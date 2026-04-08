import tensorrt as trt
import pycuda.driver as cuda
import sys

# 打印版本信息
print(f"TensorRT版本: {trt.__version__}")
print(f"CUDA版本: {cuda.get_version()}")
print(f"GPU设备数量: {cuda.Device.count()}")

# 检查GPU架构和FP16支持
if cuda.Device.count() > 0:
    dev = cuda.Device(0)
    print(f"GPU名称: {dev.name()}")
    attrs = dev.get_attributes()
    print(f"GPU计算能力: {dev.compute_capability()}")
    print(f"是否支持FP16: {trt.Builder(trt.Logger()).platform_has_fast_fp16}")
else:
    print("未检测到CUDA设备！")
