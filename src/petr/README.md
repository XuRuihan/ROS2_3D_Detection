## MultiCamDetectorNode
##本工程主要功能是在ros2节点中读取摄像头或者ros包播放的sensor_msgs::msg::CompressedImage图像topic，转化为cv::Mat格式，进行显示或保存为图片。
1. 依赖

    * ros2
    * opencv

2. 编译

    ```
    cd <path to workspace>/petr
    colcon build
    ```

`build`, `include`, `install`这三个文件夹是编译出来的，如果编译失败可以删除重新编译。

`install` 这个文件夹包含的是环境变量初始化相关的脚本，非常重要

将 `onnx` 文件编译为 TensorRT `engine` 文件：
```shell
cd models/
python3 onnx2trt.py
```


3. 运行

    ```
    source install/setup.bash
    ros2 run petr MultiCamDetector
    ```

Z:\qiyuan_project\ros2_detector_ws\src\petr\data\result里是出来的结果，

4. 播放数据包

    ```
    cd <path to rosbag>
    ros2 bag play rosbag2_2024_09_06-17_26_52
    ```
