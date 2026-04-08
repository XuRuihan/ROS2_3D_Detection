#include "trt_logger.hpp"
#include "trt_model.hpp"
#include "trt_worker.hpp"
#include "utils.hpp"

#include <cmath>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// ROS2 & CV Includes
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <builtin_interfaces/msg/time.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

// Message Filters
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

class MultiCamDetectorNode : public rclcpp::Node
{
public:
    explicit MultiCamDetectorNode(const std::string &node_name) : rclcpp::Node(node_name)
    {
        const std::string package_share_dir = ament_index_cpp::get_package_share_directory("petr");
        std::string onnxPath = package_share_dir + "/models/onnx/3dppe_v_pe.onnx";

        auto level = logger::Level::VERB;
        auto params = model::Params();

        params.img = {320, 800, 3};
        params.task = model::task_type::DETECTION;
        params.dev = model::device::GPU;
        params.prec = model::precision::FP32;
        params.n_view = 6;
        params.ws_size = 1ULL << 32;  // 4GB workspace for large PETR TensorRT builds

        // 创建一个worker的实例, 在创建的时候就完成初始化
        worker = thread::create_worker(onnxPath, level, params);
        marker_topic_ = this->declare_parameter<std::string>("marker_topic", "/petr/detections3d_markers");
        output_frame_ = this->declare_parameter<std::string>("output_frame", "base_link");
        marker_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            marker_topic_, rclcpp::QoS(10));

        current_images.resize(6);

        /* sensor_msgs::msg::CompressedImage 是压缩后的图像，需要解码 */
        std::string topics[6] = {"/FRONT_CAMERA/compressed",
                                 "/FRONT_RIGHT_CAMERA/compressed",
                                 "/FRONT_LEFT_CAMERA/compressed",
                                 "/BACK_CAMERA/compressed",
                                 "/BACK_LEFT_CAMERA/compressed",
                                 "/BACK_RIGHT_CAMERA/compressed"};

        // subscribe with message_filters and ApproximateTime sync for 6 cameras
        auto qos = rclcpp::SensorDataQoS();
        rmw_qos_profile_t custom_qos = qos.get_rmw_qos_profile();
        sub_cam0_.subscribe(this, topics[0], custom_qos);
        sub_cam1_.subscribe(this, topics[1], custom_qos);
        sub_cam2_.subscribe(this, topics[2], custom_qos);
        sub_cam3_.subscribe(this, topics[3], custom_qos);
        sub_cam4_.subscribe(this, topics[4], custom_qos);
        sub_cam5_.subscribe(this, topics[5], custom_qos);

        // 直接在 make_shared 中构造 MySyncPolicy(10)
        // 这样编译器能正确匹配到包含 Policy 参数的构造函数
        sync_ = std::make_shared<message_filters::Synchronizer<MySyncPolicy>>(
            MySyncPolicy(10), sub_cam0_, sub_cam1_, sub_cam2_, sub_cam3_, sub_cam4_, sub_cam5_);
        sync_->registerCallback(std::bind(&MultiCamDetectorNode::syncedCallback,
                                          this,
                                          std::placeholders::_1,
                                          std::placeholders::_2,
                                          std::placeholders::_3,
                                          std::placeholders::_4,
                                          std::placeholders::_5,
                                          std::placeholders::_6));
    }

    void syncedCallback(const sensor_msgs::msg::CompressedImage::ConstSharedPtr &img0,
                        const sensor_msgs::msg::CompressedImage::ConstSharedPtr &img1,
                        const sensor_msgs::msg::CompressedImage::ConstSharedPtr &img2,
                        const sensor_msgs::msg::CompressedImage::ConstSharedPtr &img3,
                        const sensor_msgs::msg::CompressedImage::ConstSharedPtr &img4,
                        const sensor_msgs::msg::CompressedImage::ConstSharedPtr &img5)
    {
        // Convert all images to cv::Mat and forward to the worker
        std::vector<cv::Mat> imgs(6);
        // 使用 toCvShare 避免深拷贝，除非你需要修改图像内容（infer通常只需要读取）
        // 如果 worker 内部会修改图像，则保持用 toCvCopy
        imgs[0] = cv::imdecode(cv::Mat(img0->data), cv::IMREAD_COLOR);
        imgs[1] = cv::imdecode(cv::Mat(img1->data), cv::IMREAD_COLOR);
        imgs[2] = cv::imdecode(cv::Mat(img2->data), cv::IMREAD_COLOR);
        imgs[3] = cv::imdecode(cv::Mat(img3->data), cv::IMREAD_COLOR);
        imgs[4] = cv::imdecode(cv::Mat(img4->data), cv::IMREAD_COLOR);
        imgs[5] = cv::imdecode(cv::Mat(img5->data), cv::IMREAD_COLOR);

        const auto detections = worker->inference_multi_and_get_result(imgs);
        publish_markers(img0->header.stamp, detections);
    }

private:
    std::string get_nuscenes_label(int label) const
    {
        static const std::vector<std::string> kNuScenesLabels = {
            "Car",
            "Truck",
            "Bus",
            "Trailer",
            "Construction Vehicle",
            "Pedestrian",
            "Motorcycle",
            "Bicycle",
            "Traffic Cone",
            "Barrier"};

        if (label < 0 || label >= static_cast<int>(kNuScenesLabels.size())) {
            return "Unknown";
        }
        return kNuScenesLabels[label];
    }

    void publish_markers(
        const builtin_interfaces::msg::Time &stamp,
        const std::vector<model::detector::bbox3d> &detections)
    {
        visualization_msgs::msg::MarkerArray marker_array;

        visualization_msgs::msg::Marker clear_marker;
        clear_marker.header.frame_id = output_frame_;
        clear_marker.header.stamp = stamp;
        clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
        marker_array.markers.push_back(clear_marker);

        int marker_id = 0;
        for (const auto &box : detections) {
            const double width = static_cast<double>(box.w);
            const double length = static_cast<double>(box.l);
            const double height = static_cast<double>(box.h);
            const double yaw = std::atan2(static_cast<double>(box.rot_sine), static_cast<double>(box.rot_cosine));
            const cv::Scalar color(
                (box.label * 53) % 255,
                (box.label * 97) % 255,
                (box.label * 193) % 255);

            visualization_msgs::msg::Marker cube_marker;
            cube_marker.header.frame_id = output_frame_;
            cube_marker.header.stamp = stamp;
            cube_marker.ns = "petr_boxes";
            cube_marker.id = marker_id++;
            cube_marker.type = visualization_msgs::msg::Marker::CUBE;
            cube_marker.action = visualization_msgs::msg::Marker::ADD;
            cube_marker.pose.position.x = box.cy;
            cube_marker.pose.position.y = -box.cx;
            cube_marker.pose.position.z = box.cz + 2.0;
            cube_marker.pose.orientation.x = 0.0;
            cube_marker.pose.orientation.y = 0.0;
            cube_marker.pose.orientation.z = std::sin(yaw * 0.5);
            cube_marker.pose.orientation.w = std::cos(yaw * 0.5);
            cube_marker.scale.x = length;
            cube_marker.scale.y = width;
            cube_marker.scale.z = height;
            cube_marker.color.a = 0.45f;
            cube_marker.color.r = static_cast<float>(color[2] / 255.0);
            cube_marker.color.g = static_cast<float>(color[1] / 255.0);
            cube_marker.color.b = static_cast<float>(color[0] / 255.0);
            cube_marker.lifetime = rclcpp::Duration::from_seconds(0.1);
            marker_array.markers.push_back(cube_marker);

            visualization_msgs::msg::Marker text_marker;
            text_marker.header.frame_id = output_frame_;
            text_marker.header.stamp = stamp;
            text_marker.ns = "petr_labels";
            text_marker.id = marker_id++;
            text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
            text_marker.action = visualization_msgs::msg::Marker::ADD;
            text_marker.pose.position.x = box.cy;
            text_marker.pose.position.y = -box.cx;
            text_marker.pose.position.z = box.cz + height * 0.6;
            text_marker.pose.orientation.w = 1.0;
            text_marker.scale.z = 0.8;
            text_marker.color.a = 1.0f;
            text_marker.color.r = 1.0f;
            text_marker.color.g = 1.0f;
            text_marker.color.b = 1.0f;
            std::ostringstream text_stream;
            text_stream << get_nuscenes_label(box.label) << " " << std::fixed << std::setprecision(2) << box.confidence;
            text_marker.text = text_stream.str();
            text_marker.lifetime = rclcpp::Duration::from_seconds(0.1);
            marker_array.markers.push_back(text_marker);
        }

        marker_publisher_->publish(marker_array);
    }

    // message_filters subscribers for each camera
    message_filters::Subscriber<sensor_msgs::msg::CompressedImage> sub_cam0_;
    message_filters::Subscriber<sensor_msgs::msg::CompressedImage> sub_cam1_;
    message_filters::Subscriber<sensor_msgs::msg::CompressedImage> sub_cam2_;
    message_filters::Subscriber<sensor_msgs::msg::CompressedImage> sub_cam3_;
    message_filters::Subscriber<sensor_msgs::msg::CompressedImage> sub_cam4_;
    message_filters::Subscriber<sensor_msgs::msg::CompressedImage> sub_cam5_;
    using MySyncPolicy = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::CompressedImage, sensor_msgs::msg::CompressedImage,
        sensor_msgs::msg::CompressedImage, sensor_msgs::msg::CompressedImage,
        sensor_msgs::msg::CompressedImage, sensor_msgs::msg::CompressedImage>;

    std::shared_ptr<message_filters::Synchronizer<MySyncPolicy>> sync_;

    std::vector<cv::Mat> current_images;
    std::shared_ptr<thread::Worker> worker;
    std::string marker_topic_;
    std::string output_frame_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_publisher_;
};

int main(int argc, char const *argv[])
{
    /*这么实现目的在于让调用的整个过程精简化*/
    rclcpp::init(argc, argv);
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 1, true);
    auto node = std::make_shared<MultiCamDetectorNode>("MultiCamDetector");
    executor.add_node(node);
    executor.spin();
    // rclcpp::shutdown();
    return 0;
}
