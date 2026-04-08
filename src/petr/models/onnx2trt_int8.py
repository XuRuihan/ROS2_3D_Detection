import os
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # 自动初始化CUDA上下文


# 校准器实现（Int8EntropyCalibrator2）
class Int8EntropyCalibrator2(trt.IInt8EntropyCalibrator2):
    def __init__(
        self,
        calib_data_dir,
        input_shape,
        batch_size=1,
        cache_file="int8_calibration.cache",
    ):
        """
        初始化INT8熵校准器
        :param calib_data_dir: 校准数据集目录（存放npy格式的输入数据）
        :param input_shape: 模型输入形状 (batch_size, channels, height, width)
        :param batch_size: 校准批大小
        :param cache_file: 校准缓存文件路径
        """
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.current_index = 0

        # 加载校准数据集
        self.calib_data = self.load_calibration_data(calib_data_dir)
        self.num_batches = len(self.calib_data) // self.batch_size

        # 分配设备内存
        self.device_input = cuda.mem_alloc(self.calib_data[0].nbytes * self.batch_size)

    def load_calibration_data(self, data_dir):
        """加载校准数据（npy文件）"""
        calib_files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".npy")
        ]
        calib_data = []
        for file in calib_files:
            data = np.load(file).astype(np.float32)
            # 确保数据形状匹配
            if data.shape == self.input_shape[1:]:  # 排除batch维度
                calib_data.append(data)
            else:
                print(
                    f"跳过不匹配的数据文件 {file}，形状: {data.shape}, 期望: {self.input_shape[1:]}"
                )
        return np.array(calib_data)

    def get_batch_size(self):
        """返回批大小"""
        return self.batch_size

    def get_batch(self, names):
        """获取下一批校准数据（必须实现）"""
        if self.current_index + self.batch_size > len(self.calib_data):
            return None  # 没有更多数据

        # 获取当前批次数据
        batch = self.calib_data[
            self.current_index : self.current_index + self.batch_size
        ]
        self.current_index += self.batch_size

        # 将数据拷贝到设备内存
        cuda.memcpy_htod(self.device_input, batch.ravel())
        return [int(self.device_input)]

    def read_calibration_cache(self):
        """读取校准缓存（加速后续转换）"""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        """保存校准缓存"""
        with open(self.cache_file, "wb") as f:
            f.write(cache)


# ONNX转TensorRT Engine（INT8）
def onnx_to_tensorrt_int8(
    onnx_file,
    engine_file,
    calib_data_dir,
    input_shape=(1, 3, 224, 224),
    batch_size=1,
    max_workspace_size=1 << 30,
):
    """
    将ONNX模型转换为INT8精度的TensorRT Engine
    :param onnx_file: 输入ONNX文件路径
    :param engine_file: 输出Engine文件路径
    :param calib_data_dir: 校准数据集目录
    :param input_shape: 模型输入形状 (batch, channel, h, w)
    :param batch_size: 批大小
    :param max_workspace_size: 最大工作空间大小（默认1GB）
    """
    # 初始化TensorRT logger
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)

    # 创建构建器和网络
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # 解析ONNX文件
    with open(onnx_file, "rb") as f:
        if not parser.parse(f.read()):
            print("ONNX模型解析失败:")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False

    # 创建构建配置
    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace_size

    # 设置INT8模式
    config.set_flag(trt.BuilderFlag.INT8)
    # 创建校准器
    calibrator = Int8EntropyCalibrator2(
        calib_data_dir=calib_data_dir,
        input_shape=input_shape,
        batch_size=batch_size,
        cache_file="int8_calib.cache",
    )
    config.int8_calibrator = calibrator

    # 设置最大批大小
    builder.max_batch_size = batch_size

    # 构建并序列化Engine
    print("开始构建INT8 TensorRT Engine...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("Engine构建失败！")
        return False

    # 保存Engine文件
    with open(engine_file, "wb") as f:
        f.write(serialized_engine)
    print(f"INT8 Engine文件已保存至: {engine_file}")
    return True


# 主函数
if __name__ == "__main__":
    # 配置参数
    ONNX_FILE = "onnx/3dppe.onnx"  # 你的ONNX模型路径
    ENGINE_FILE = "engine/3dppe.engine"  # 输出的Engine文件路径
    CALIB_DATA_DIR = "./calib_data"  # 校准数据集目录（存放npy文件）
    INPUT_SHAPE = (1, 3, 224, 224)  # 模型输入形状 (batch, channel, h, w)
    BATCH_SIZE = 1  # 校准批大小

    # 执行转换
    success = onnx_to_tensorrt_int8(
        onnx_file=ONNX_FILE,
        engine_file=ENGINE_FILE,
        calib_data_dir=CALIB_DATA_DIR,
        input_shape=INPUT_SHAPE,
        batch_size=BATCH_SIZE,
    )

    if success:
        print("ONNX转INT8 TensorRT Engine成功！")
    else:
        print("ONNX转INT8 TensorRT Engine失败！")
