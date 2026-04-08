import argparse
import os
import tensorrt as trt


def onnx_to_tensorrt(
    onnx_file, engine_file, precision="fp32", max_workspace_size=1 << 32
):
    """
    将ONNX模型转换为FP32/FP16精度的TensorRT Engine（无需校准数据）
    :param onnx_file: 输入ONNX文件路径
    :param engine_file: 输出Engine文件路径
    :param precision: 精度模式，可选 "fp32" 或 "fp16"
    :param max_workspace_size: 最大工作空间大小（默认1GB）
    """
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)

    # 创建构建器、网络和解析器
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # 解析ONNX文件
    with open(onnx_file, "rb") as f:
        if not parser.parse(f.read()):
            print("ONNX解析失败:")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False

    # 配置构建参数
    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace_size

    # 设置精度模式
    if precision == "fp16" and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("启用FP16精度模式")
    elif precision == "fp16" and not builder.platform_has_fast_fp16:
        print("当前设备不支持FP16，自动切换为FP32")

    # 构建并保存Engine
    print(f"开始构建{precision.upper()}精度Engine...")
    serialized_engine = builder.build_serialized_network(network, config)
    if not serialized_engine:
        print("Engine构建失败！")
        return False

    with open(engine_file, "wb") as f:
        f.write(serialized_engine)
    print(f"{precision.upper()} Engine已保存至: {engine_file}")
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Convert ONNX to TensorRT engine.")
    parser.add_argument("--onnx", default="onnx/3dppe_v_pe.onnx", help="Path to input ONNX.")
    parser.add_argument(
        "--engine",
        default="engine/3dppe_v_pe-fp32.engine",
        help="Path to output TensorRT engine.",
    )
    parser.add_argument(
        "--precision",
        default="fp32",
        choices=["fp32", "fp16"],
        help="TensorRT build precision.",
    )
    parser.add_argument(
        "--workspace",
        type=int,
        default=1 << 32,
        help="Max TensorRT workspace size in bytes.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    onnx_to_tensorrt(
        args.onnx,
        args.engine,
        precision=args.precision,
        max_workspace_size=args.workspace,
    )
