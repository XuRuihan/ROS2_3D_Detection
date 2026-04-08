cmake_minimum_required(VERSION 3.14)
project(petr VERSION 0.0.1 LANGUAGES CXX CUDA)

# if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#   add_compile_options(-Wall -Wextra -Wpedantic)
# endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_ARCHITECTURES 80)

# BUILD TYPE
set(CMAKE_BUILD_TYPE RELEASE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 设置项目名称变量
set(PARENT_PROJECT_NAME ${PROJECT_NAME})

# TensorRT
find_path(TensorRT_INCLUDE_DIR
  NAMES NvInfer.h
  PATHS /usr/include/x86_64-linux-gnu /usr/include /usr/local/include /opt/TensorRT/include /opt/TensorRT-8.5.3.1/include
)
find_library(TensorRT_NVINFER_LIB
  NAMES nvinfer
  PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib /opt/TensorRT/lib /opt/TensorRT-8.5.3.1/lib
)
find_library(TensorRT_NVONNXPARSER_LIB
  NAMES nvonnxparser
  PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib /opt/TensorRT/lib /opt/TensorRT-8.5.3.1/lib
)

if(NOT TensorRT_INCLUDE_DIR)
  message(FATAL_ERROR "TensorRT header NvInfer.h not found")
endif()
if(NOT TensorRT_NVINFER_LIB)
  message(FATAL_ERROR "TensorRT library nvinfer not found")
endif()
if(NOT TensorRT_NVONNXPARSER_LIB)
  message(FATAL_ERROR "TensorRT library nvonnxparser not found")
endif()

get_filename_component(TensorRT_LIB_DIRS ${TensorRT_NVINFER_LIB} DIRECTORY)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CURL REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(yaml-cpp REQUIRED)

option(PETR_ENABLE_LINT "Enable ament lint checks during BUILD_TESTING" OFF)

if(BUILD_TESTING AND PETR_ENABLE_LINT)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

add_executable(${PROJECT_NAME} 
    src/cpp/main.cpp
    src/cpp/trt_calibrator.cpp
    src/cpp/trt_classifier.cpp
    src/cpp/trt_detector.cpp
    src/cpp/trt_logger.cpp
    src/cpp/trt_model.cpp
    src/cpp/trt_preprocess.cpp
    src/cpp/trt_preprocess.cu
    src/cpp/trt_timer.cpp
    src/cpp/trt_worker.cpp
    src/cpp/utils.cpp
)

set_source_files_properties(src/cpp/trt_preprocess.cu PROPERTIES CUDA_STANDARD 14)

ament_target_dependencies(${PROJECT_NAME}
  ament_index_cpp
  rclcpp
  std_msgs
  sensor_msgs
  visualization_msgs
  message_filters
  cv_bridge
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_directories(${PROJECT_NAME} PRIVATE
    ${TensorRT_LIB_DIRS}
)

target_link_libraries(${PROJECT_NAME}
    pthread
    cuda
    cudart
    cudnn
    ${TensorRT_NVINFER_LIB}
    ${TensorRT_NVONNXPARSER_LIB}
    ${OpenCV_LIBRARIES}
    ${CUDA_LIBRARIES}
    CURL::libcurl
    yaml-cpp
)

install(TARGETS
  ${PROJECT_NAME}
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  models
  calibration
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
