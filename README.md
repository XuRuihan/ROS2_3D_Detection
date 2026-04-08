# ROS2 3D Detection Workspace

This repository contains a ROS 2 workspace for camera-based 3D object detection with TensorRT acceleration. The main package, `petr`, runs a multi-camera PETR-style detector inside a ROS 2 node, consumes 6 synchronized compressed camera streams, performs TensorRT inference, and publishes both 3D markers and image-space visualizations.

The current implementation is designed around a 6-view setup similar to nuScenes. It loads camera calibration from YAML, prepares PETR auxiliary tensors such as `intrinsics`, `extrinsics`, and `img2lidar`, runs inference on resized `320 x 800` inputs, and publishes:

- 3D bounding boxes as `visualization_msgs/msg/MarkerArray`
- 2D projected 3D boxes on per-camera image topics
- TensorRT-based inference from ONNX or serialized engine files

## What This Project Does

At a high level, this workspace provides:

- A ROS 2 executable `petr` built from the package in [`src/petr`](/nfs/ruihanxu/qiyuan_project/ros2_detector_ws/src/petr)
- A multi-camera node named `MultiCamDetector`
- TensorRT model loading and engine building utilities
- Calibration handling for multi-view camera geometry
- 3D detection visualization in both RViz and image space

The runtime flow is:

1. Subscribe to six `sensor_msgs/msg/CompressedImage` camera topics.
2. Synchronize them with `message_filters::Synchronizer`.
3. Decode each image into `cv::Mat`.
4. Prepare PETR inputs:
   - image tensor
   - camera intrinsics
   - camera extrinsics
   - `img2lidar`
5. Run TensorRT inference.
6. Decode 3D detections.
7. Publish:
   - 3D markers to RViz
   - projected 3D boxes on resized `320 x 800` visualization images

## Repository Layout

Important files and directories:

- [`src/petr`](/nfs/ruihanxu/qiyuan_project/ros2_detector_ws/src/petr): main ROS 2 package
- [`src/petr/src/cpp/main.cpp`](/nfs/ruihanxu/qiyuan_project/ros2_detector_ws/src/petr/src/cpp/main.cpp): multi-camera ROS 2 node
- [`src/petr/src/cpp/trt_detector.cpp`](/nfs/ruihanxu/qiyuan_project/ros2_detector_ws/src/petr/src/cpp/trt_detector.cpp): PETR TensorRT detector implementation
- [`src/petr/src/cpp/trt_worker.cpp`](/nfs/ruihanxu/qiyuan_project/ros2_detector_ws/src/petr/src/cpp/trt_worker.cpp): inference worker wrapper
- [`src/petr/config`](/nfs/ruihanxu/qiyuan_project/ros2_detector_ws/src/petr/config): sensor calibration and scene YAML files
- [`src/petr/models`](/nfs/ruihanxu/qiyuan_project/ros2_detector_ws/src/petr/models): ONNX/TensorRT conversion scripts and model assets
- [`src/petr/tools`](/nfs/ruihanxu/qiyuan_project/ros2_detector_ws/src/petr/tools): helper scripts for building and profiling TensorRT engines

## Requirements

This project is intended for Linux with NVIDIA GPU support.

### Core dependencies

- ROS 2 Humble
- CMake 3.14+
- GCC/G++ with C++17 support
- CUDA Toolkit
- cuDNN
- TensorRT
- OpenCV
- `cv_bridge`
- `yaml-cpp`
- `message_filters`

### ROS 2 package dependencies

The `petr` package depends on (could be installed with rosdep later):

- `ament_cmake`
- `ament_index_cpp`
- `rclcpp`
- `std_msgs`
- `sensor_msgs`
- `visualization_msgs`
- `message_filters`
- `cv_bridge`
- `yaml-cpp`

## Environment Setup

### 1. Install ROS 2 Humble

Install ROS 2 Humble first and make sure the environment can be sourced:

```bash
source /opt/ros/humble/setup.bash
```

If you prefer installing ROS with `fishros`, you can use the following sequence:

```bash
wget http://fishros.com/install -O fishros
chmod 777 ./fishros
printf "1\n1\n2\n1\n5\n1\n1" | ./fishros
source /opt/ros/humble/setup.bash
```

or you can install with customized options with original `fishros`:

```bash
wget http://fishros.com/install -O fishros && . fishros
```

### 2. Install system packages

You can install dependencies either manually with `apt`, or automatically with `rosdep`. For most ROS 2 users, `rosdep` is the recommended way because it resolves package dependencies directly from the workspace. For users in mainland China, `rosdepc` is a practical alternative that works similarly but is often more reliable with domestic mirrors.

### Option A: install common packages manually

Install the common development packages used by this workspace:

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  git \
  libopencv-dev \
  libyaml-cpp-dev \
  ros-humble-cv-bridge \
  ros-humble-message-filters \
  ros-humble-vision-msgs \
  python3-colcon-common-extensions
```

If `ros-humble-vision-msgs` is not needed in your local setup, it can be omitted.

### Option B: install dependencies with `rosdep`

From the workspace root:

```bash
cd /path/to/ros2_detector_ws
source /opt/ros/humble/setup.bash
sudo rosdep init
rosdep update
rosdep install --from-paths src --ignore-src -r -y
```

This command scans the packages under `src/` and installs the missing system dependencies required to build them.

### Option C: install dependencies with `rosdepc` for users in China

If you are in mainland China, `rosdepc` may provide a smoother dependency installation experience:

```bash
sudo apt update
sudo apt install -y python3-pip
python3 -m pip install -U rosdepc
sudo rosdepc init
rosdepc update
cd /path/to/ros2_detector_ws
source /opt/ros/humble/setup.bash
rosdepc install --from-paths src --ignore-src -r -y
```

`rosdepc` is used in the same way as `rosdep`, so you can usually substitute one for the other in dependency installation commands.

If `rosdep` or `rosdepc` has already been initialized on your machine, you can skip the corresponding `init` step.

### 3. Install CUDA, cuDNN, and TensorRT

Make sure the following libraries are available on your machine:

- `cuda`
- `cudart`
- `cudnn`
- `nvinfer`
- `nvonnxparser`

The package CMake currently searches typical TensorRT install locations such as:

- `/usr/include/x86_64-linux-gnu`
- `/usr/lib/x86_64-linux-gnu`
- `/opt/TensorRT/include`
- `/opt/TensorRT/lib`

If your TensorRT installation is in a custom path, update [`src/petr/CMakeLists.txt`](/nfs/ruihanxu/qiyuan_project/ros2_detector_ws/src/petr/CMakeLists.txt) accordingly.

### 4. (Optional) Python tools for model conversion

If you want to build TensorRT engines from ONNX using the provided scripts, install Python packages required by your TensorRT Python environment.

Typical setup:

```bash
python3 -m pip install --upgrade pip numpy
```

You also need TensorRT Python bindings if you plan to use:

- [`src/petr/models/onnx2trt.py`](/nfs/ruihanxu/qiyuan_project/ros2_detector_ws/src/petr/models/onnx2trt.py)
- [`src/petr/models/onnx2trt_int8.py`](/nfs/ruihanxu/qiyuan_project/ros2_detector_ws/src/petr/models/onnx2trt_int8.py)

## Docker Setup

This repository also provides a Docker environment definition at:

- [`docker/Dockerfile`](/nfs/ruihanxu/qiyuan_project/ros2_detector_ws/docker/Dockerfile)

The Docker image is based on:

- `nvidia/cuda:11.4.3-devel-ubuntu20.04`

It installs:

- Python 3.8
- cuDNN 8.6
- TensorRT 8.5.3
- ROS via `fishros`

### Build the Docker image

From the workspace root:

```bash
cd /path/to/ros2_detector_ws
docker build -f docker/Dockerfile -t ros2-3d-detection:cu114 .
```

### Run the container

If you want GPU access, use the NVIDIA runtime:

```bash
docker run --gpus all -it --rm \
  --network host \
  -v /path/to/ros2_detector_ws:/workspace/ros2_detector_ws \
  ros2-3d-detection:cu114
```

Inside the container:

```bash
cd /workspace/ros2_detector_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select petr
source install/setup.bash
ros2 run petr petr
```

### Notes for Docker users

- Make sure the host has a working NVIDIA driver installation.
- Make sure Docker supports `--gpus all`.
- If you need GUI tools such as RViz, you may also need X11 forwarding or an equivalent display setup.
- If your camera streams or ROS graph run on the host, `--network host` is usually the simplest option.

## Build Instructions

From the workspace root:

```bash
cd /path/to/ros2_detector_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select petr
```

After a successful build:

```bash
source install/setup.bash
```

If you want to rebuild from scratch:

```bash
rm -rf build install log
colcon build --packages-select petr
```

or

```bash
colcon build --packages-select petr --cmake-clean-cache
```

## Model Preparation

The node expects the ONNX model at:

[`src/petr/models/onnx/3dppe_v_pe.onnx`](/nfs/ruihanxu/qiyuan_project/ros2_detector_ws/src/petr/models/onnx/3dppe_v_pe.onnx)

Instructions for obtaining the ONNX file will be provided in the following repository:

[https://github.com/XuRuihan/LightPerception](https://github.com/XuRuihan/LightPerception)

At runtime, the code will try to load a TensorRT engine. If the engine does not exist, it attempts to build one automatically.

You can also generate an engine manually.

### Convert ONNX to TensorRT with Python

```bash
cd src/petr/models
python3 onnx2trt.py \
  --onnx onnx/3dppe_v_pe.onnx \
  --engine engine/3dppe_v_pe-fp32.engine \
  --precision fp32
```

### Convert ONNX to TensorRT with `trtexec`

```bash
cd src/petr/tools
./build.sh
```

<!-- ### Profile an existing engine

```bash
cd src/petr/tools
./profile.sh /path/to/model.engine profile_tag
``` -->

## Calibration and Sensor Metadata

The detector requires per-camera calibration metadata. Example files are provided in:

- [`src/petr/config/nuscenes_sensor_yaml`](/nfs/ruihanxu/qiyuan_project/ros2_detector_ws/src/petr/config/nuscenes_sensor_yaml)

The node reads camera metadata such as:

- `intrinsics`
- `cam2lidar`
- `lidar2cam`
- `lidar2img`
- `img2lidar`

The default runtime configuration is:

- input image size: `320 x 800`
- number of cameras: `6`
- default sensor config:
  [`src/petr/config/nuscenes_sensor_yaml/scene-0061.yaml`](/nfs/ruihanxu/qiyuan_project/ros2_detector_ws/src/petr/config/nuscenes_sensor_yaml/scene-0061.yaml)

## Running the Node

After building and sourcing the workspace:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run petr petr
```

The executable name is `petr`, while the ROS 2 node name created in code is `MultiCamDetector`.

### Run with a custom sensor calibration file

You can pass a calibration file from the command line:

```bash
ros2 run petr petr --sensor-config /absolute/path/to/scene-0916.yaml
```

You can also use ROS parameters:

```bash
ros2 run petr petr --ros-args \
  -p use_sim_time:=true
  -p sensor_info_path:=/absolute/path/to/scene-0916.yaml \
  -p marker_topic:=/petr/detections3d_markers \
  -p output_frame:=base_link
```

## Input Topics

The current node subscribes to these six compressed image topics:

- `/FRONT_CAMERA/compressed`
- `/FRONT_RIGHT_CAMERA/compressed`
- `/FRONT_LEFT_CAMERA/compressed`
- `/BACK_CAMERA/compressed`
- `/BACK_LEFT_CAMERA/compressed`
- `/BACK_RIGHT_CAMERA/compressed`

These topics are synchronized with approximate time policy before inference.

## Output Topics

### 3D markers

By default, 3D detections are published as:

- `/petr/detections3d_markers`

Message type:

- `visualization_msgs/msg/MarkerArray`

### Per-camera visualization images

For each subscribed compressed image topic, the node publishes a visualization image topic with the suffix `/objects_3d`.

Examples:

- `/FRONT_CAMERA/objects_3d`
- `/FRONT_RIGHT_CAMERA/objects_3d`
- `/FRONT_LEFT_CAMERA/objects_3d`
- `/BACK_CAMERA/objects_3d`
- `/BACK_LEFT_CAMERA/objects_3d`
- `/BACK_RIGHT_CAMERA/objects_3d`

These images contain projected 3D bounding boxes drawn on resized `320 x 800` images.

## Running with a ROS 2 Bag

If your six camera topics are recorded in a bag:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 bag play /path/to/your_bag
```

Then in another terminal:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run petr petr
```

A more handy bag play is:
```bash
ros2 bag play -l /path/to/your_bag --read-ahead-queue-size 100 --clock
```
where `-l` plays the bag with loop, `--read-ahead-queue-size` accelerates the bag loading, and `--clock` synchronizes the timestamp during debugging.

## Visualization

### RViz

To inspect 3D detections in RViz:

1. Open RViz 2.
2. Set the fixed frame to match `output_frame` (default: `base_link`).
3. Add a `MarkerArray` display.
4. Subscribe it to `/petr/detections3d_markers`.

You can start RViz 2 with

```bash
rviz2
```

or you can start RViz 2 with simulated timestamp:

```bash
ros2 run rviz2 rviz2 --ros-args -p use_sim_time:=true
```

### Image overlays

To inspect projected 3D boxes on images:

```bash
ros2 topic list | grep objects_3d
```

Then visualize the published image topics with your preferred ROS image viewer.

## Notes and Known Assumptions

- The current package is configured for a six-camera PETR-like setup.
- The TensorRT detector currently expects four input tensors:
  - images
  - intrinsics
  - extrinsics
  - `img2lidar`
- The detector currently handles five output tensors, including decoded outputs used for postprocessing.
- The visualization path is tuned for nuScenes-style calibration content provided in the YAML files.
- The build configuration uses CUDA architecture `80` by default. If your GPU differs, update [`src/petr/CMakeLists.txt`](/nfs/ruihanxu/qiyuan_project/ros2_detector_ws/src/petr/CMakeLists.txt).

## Citation

If this repository contributes to your research or internal project, please cite the original PETR-related work and any upstream detector/model source you used to export the ONNX model.

A generic BibTeX placeholder:

```bibtex
@misc{ros2_3d_detection,
  title  = {ROS2 3D Detection},
  author = {Ruihan Xu},
  year   = {2026},
  note   = {ROS 2 multi-camera 3D detection demo}
}
```

If you use the official PETR model or codebase, please also cite the original PETR paper in your final project documentation.

## Acknowledgements

This workspace builds on ideas and tooling from several open-source communities:

- ROS 2 for runtime middleware and visualization interfaces
- OpenCV for image decoding and drawing
- NVIDIA CUDA and TensorRT for accelerated inference
- PETR-style multi-view 3D detection research and deployment workflows
- The broader autonomous driving and nuScenes ecosystem for multi-camera calibration conventions

Thanks to the contributors who prepared the calibration files, TensorRT wrappers, ROS 2 integration, and visualization pipeline in this workspace.
