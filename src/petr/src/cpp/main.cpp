#include "trt_logger.hpp"
#include "trt_model.hpp"
#include "trt_worker.hpp"
#include "utils.hpp"

#include <cmath>
#include <array>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <limits>
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
#include <yaml-cpp/yaml.h>

// Message Filters
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

class MultiCamDetectorNode : public rclcpp::Node
{
public:
    static constexpr std::size_t kCameraCount = 6;

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
        sensor_info_path_ = this->declare_parameter<std::string>(
            "sensor_info_path", package_share_dir + "/config/sensor_info_nuscenes.yaml");
        marker_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            marker_topic_, rclcpp::QoS(10));

        current_images.resize(kCameraCount);
        camera_names_ = {
            "FRONT_CAMERA",
            "FRONT_RIGHT_CAMERA",
            "FRONT_LEFT_CAMERA",
            "BACK_CAMERA",
            "BACK_LEFT_CAMERA",
            "BACK_RIGHT_CAMERA"};
        image_topics_ = {
            "/FRONT_CAMERA/compressed",
            "/FRONT_RIGHT_CAMERA/compressed",
            "/FRONT_LEFT_CAMERA/compressed",
            "/BACK_CAMERA/compressed",
            "/BACK_LEFT_CAMERA/compressed",
            "/BACK_RIGHT_CAMERA/compressed"};
        projection_matrices_.resize(kCameraCount);
        has_projection_matrix_.assign(kCameraCount, false);
        image_publishers_.resize(kCameraCount);
        load_projection_matrices_from_yaml(sensor_info_path_);

        for (std::size_t camera_index = 0; camera_index < kCameraCount; ++camera_index) {
            if (!has_projection_matrix_[camera_index]) {
                RCLCPP_WARN(
                    this->get_logger(),
                    "Projection matrix for %s is not available from %s. Publish raw image only on visualization topic.",
                    camera_names_[camera_index].c_str(),
                    sensor_info_path_.c_str());
            }

            image_publishers_[camera_index] = this->create_publisher<sensor_msgs::msg::Image>(
                make_visualization_topic(image_topics_[camera_index]), rclcpp::SensorDataQoS());
        }

        // subscribe with message_filters and ApproximateTime sync for 6 cameras
        auto qos = rclcpp::SensorDataQoS();
        rmw_qos_profile_t custom_qos = qos.get_rmw_qos_profile();
        sub_cam0_.subscribe(this, image_topics_[0], custom_qos);
        sub_cam1_.subscribe(this, image_topics_[1], custom_qos);
        sub_cam2_.subscribe(this, image_topics_[2], custom_qos);
        sub_cam3_.subscribe(this, image_topics_[3], custom_qos);
        sub_cam4_.subscribe(this, image_topics_[4], custom_qos);
        sub_cam5_.subscribe(this, image_topics_[5], custom_qos);

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
        publish_visualizations({img0, img1, img2, img3, img4, img5}, imgs, detections);
    }

private:
    using ProjectionMatrix = cv::Matx34d;

    static std::string make_visualization_topic(const std::string &image_topic)
    {
        constexpr char kCompressedSuffix[] = "/compressed";
        constexpr char kVisualizationSuffix[] = "/objects_3d";
        if (image_topic.size() >= sizeof(kCompressedSuffix) - 1 &&
            image_topic.compare(
                image_topic.size() - (sizeof(kCompressedSuffix) - 1),
                sizeof(kCompressedSuffix) - 1,
                kCompressedSuffix) == 0) {
            return image_topic.substr(0, image_topic.size() - (sizeof(kCompressedSuffix) - 1)) + kVisualizationSuffix;
        }
        return image_topic + kVisualizationSuffix;
    }

    bool load_projection_matrix(const std::vector<double> &matrix_data, ProjectionMatrix &projection_matrix) const
    {
        if (matrix_data.empty()) {
            return false;
        }

        if (matrix_data.size() != 12) {
            RCLCPP_ERROR(
                this->get_logger(),
                "Projection matrix parameter must contain 12 elements, but got %zu.",
                matrix_data.size());
            return false;
        }

        projection_matrix = ProjectionMatrix(
            matrix_data[0], matrix_data[1], matrix_data[2], matrix_data[3],
            matrix_data[4], matrix_data[5], matrix_data[6], matrix_data[7],
            matrix_data[8], matrix_data[9], matrix_data[10], matrix_data[11]);
        return true;
    }

    void load_projection_matrices_from_yaml(const std::string &yaml_path)
    {
        try {
            const YAML::Node root = YAML::LoadFile(yaml_path);
            const YAML::Node camera_nodes = root["Camera"];
            if (!camera_nodes || !camera_nodes.IsSequence()) {
                RCLCPP_ERROR(this->get_logger(), "Invalid sensor info yaml: missing Camera sequence in %s.", yaml_path.c_str());
                return;
            }

            for (const auto &camera_node : camera_nodes) {
                const YAML::Node name_node = camera_node["name"];
                const YAML::Node lidar2img_node = camera_node["lidar2img"];
                if (!name_node || !lidar2img_node || !lidar2img_node.IsSequence() || lidar2img_node.size() < 3) {
                    continue;
                }

                const std::string camera_name = name_node.as<std::string>();
                auto camera_iter = std::find(camera_names_.begin(), camera_names_.end(), camera_name);
                if (camera_iter == camera_names_.end()) {
                    continue;
                }

                std::vector<double> projection_data;
                projection_data.reserve(12);
                bool valid_matrix = true;
                for (std::size_t row = 0; row < 3 && valid_matrix; ++row) {
                    const YAML::Node row_node = lidar2img_node[row];
                    if (!row_node || !row_node.IsSequence() || row_node.size() < 4) {
                        valid_matrix = false;
                        break;
                    }
                    for (std::size_t col = 0; col < 4; ++col) {
                        projection_data.push_back(row_node[col].as<double>());
                    }
                }

                if (!valid_matrix) {
                    RCLCPP_ERROR(
                        this->get_logger(),
                        "Invalid lidar2img matrix for %s in %s.",
                        camera_name.c_str(),
                        yaml_path.c_str());
                    continue;
                }

                const std::size_t camera_index = static_cast<std::size_t>(std::distance(camera_names_.begin(), camera_iter));
                has_projection_matrix_[camera_index] = load_projection_matrix(projection_data, projection_matrices_[camera_index]);
            }
        } catch (const YAML::Exception &e) {
            RCLCPP_ERROR(
                this->get_logger(),
                "Failed to load sensor info yaml %s: %s",
                yaml_path.c_str(),
                e.what());
        }
    }

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

    cv::Scalar get_label_color(int label) const
    {
        return cv::Scalar(
            (label * 53) % 255,
            (label * 97) % 255,
            (label * 193) % 255);
    }

    cv::Point3d get_box_center_in_output_frame(const model::detector::bbox3d &box) const
    {
        return cv::Point3d(
            static_cast<double>(box.cx),
            static_cast<double>(box.cy),
            static_cast<double>(box.cz));
    }

    std::array<cv::Point3d, 8> get_box_corners_in_output_frame(const model::detector::bbox3d &box) const
    {
        const double half_length = static_cast<double>(box.l) * 0.5;
        const double half_width = static_cast<double>(box.w) * 0.5;
        const double half_height = static_cast<double>(box.h) * 0.5;
        const double yaw = std::atan2(static_cast<double>(box.rot_sine), static_cast<double>(box.rot_cosine));
        const double cos_yaw = std::cos(yaw);
        const double sin_yaw = std::sin(yaw);
        const cv::Point3d center = get_box_center_in_output_frame(box);

        const std::array<cv::Point3d, 8> local_corners = {
            cv::Point3d( half_length,  half_width,  half_height),
            cv::Point3d( half_length, -half_width,  half_height),
            cv::Point3d(-half_length, -half_width,  half_height),
            cv::Point3d(-half_length,  half_width,  half_height),
            cv::Point3d( half_length,  half_width, -half_height),
            cv::Point3d( half_length, -half_width, -half_height),
            cv::Point3d(-half_length, -half_width, -half_height),
            cv::Point3d(-half_length,  half_width, -half_height)};

        std::array<cv::Point3d, 8> corners;
        for (std::size_t i = 0; i < local_corners.size(); ++i) {
            const auto &point = local_corners[i];
            corners[i] = cv::Point3d(
                center.x + cos_yaw * point.x - sin_yaw * point.y,
                center.y + sin_yaw * point.x + cos_yaw * point.y,
                center.z + point.z);
        }
        return corners;
    }

    bool project_point(
        const ProjectionMatrix &projection_matrix,
        const cv::Point3d &point_3d,
        cv::Point2d &point_2d) const
    {
        const cv::Vec4d point_h(point_3d.x, point_3d.y, point_3d.z, 1.0);
        const cv::Vec3d projected = projection_matrix * point_h;
        if (projected[2] <= 1e-6) {
            return false;
        }

        point_2d.x = projected[0] / projected[2];
        point_2d.y = projected[1] / projected[2];
        return std::isfinite(point_2d.x) && std::isfinite(point_2d.y);
    }

    void draw_projected_box(
        cv::Mat &image,
        const ProjectionMatrix &projection_matrix,
        const model::detector::bbox3d &box) const
    {
        static const std::array<std::pair<int, int>, 12> kEdges = {
            std::pair<int, int>(0, 1), std::pair<int, int>(1, 2), std::pair<int, int>(2, 3), std::pair<int, int>(3, 0),
            std::pair<int, int>(4, 5), std::pair<int, int>(5, 6), std::pair<int, int>(6, 7), std::pair<int, int>(7, 4),
            std::pair<int, int>(0, 4), std::pair<int, int>(1, 5), std::pair<int, int>(2, 6), std::pair<int, int>(3, 7)};

        const auto corners = get_box_corners_in_output_frame(box);
        std::array<cv::Point2d, 8> projected_corners;
        std::array<bool, 8> valid_corners{};
        int valid_count = 0;
        double min_x = std::numeric_limits<double>::max();
        double min_y = std::numeric_limits<double>::max();

        for (std::size_t i = 0; i < corners.size(); ++i) {
            valid_corners[i] = project_point(projection_matrix, corners[i], projected_corners[i]);
            if (valid_corners[i]) {
                ++valid_count;
                min_x = std::min(min_x, projected_corners[i].x);
                min_y = std::min(min_y, projected_corners[i].y);
            }
        }

        if (valid_count < 4) {
            return;
        }

        const cv::Scalar color = get_label_color(box.label);
        for (const auto &[start_index, end_index] : kEdges) {
            if (!valid_corners[start_index] || !valid_corners[end_index]) {
                continue;
            }

            cv::line(
                image,
                projected_corners[start_index],
                projected_corners[end_index],
                color,
                2,
                cv::LINE_AA);
        }

        std::ostringstream text_stream;
        text_stream << get_nuscenes_label(box.label) << " " << std::fixed << std::setprecision(2) << box.confidence;
        cv::putText(
            image,
            text_stream.str(),
            cv::Point(
                static_cast<int>(std::max(0.0, min_x)),
                static_cast<int>(std::max(20.0, min_y - 6.0))),
            cv::FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv::LINE_AA);
    }

    void publish_visualizations(
        const std::array<sensor_msgs::msg::CompressedImage::ConstSharedPtr, kCameraCount> &compressed_images,
        const std::vector<cv::Mat> &decoded_images,
        const std::vector<model::detector::bbox3d> &detections)
    {
        for (std::size_t camera_index = 0; camera_index < kCameraCount; ++camera_index) {
            if (decoded_images[camera_index].empty()) {
                RCLCPP_WARN_THROTTLE(
                    this->get_logger(), *this->get_clock(), 2000,
                    "Decoded image is empty for %s, skip visualization publish.",
                    camera_names_[camera_index].c_str());
                continue;
            }

            cv::Mat visualized_image = decoded_images[camera_index].clone();
            if (has_projection_matrix_[camera_index]) {
                for (const auto &box : detections) {
                    draw_projected_box(visualized_image, projection_matrices_[camera_index], box);
                }
            }

            auto image_msg = cv_bridge::CvImage(
                compressed_images[camera_index]->header,
                sensor_msgs::image_encodings::BGR8,
                visualized_image).toImageMsg();
            image_publishers_[camera_index]->publish(*image_msg);
        }
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
            const cv::Scalar color = get_label_color(box.label);

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
    std::array<std::string, kCameraCount> camera_names_;
    std::array<std::string, kCameraCount> image_topics_;
    std::vector<ProjectionMatrix> projection_matrices_;
    std::vector<bool> has_projection_matrix_;
    std::string marker_topic_;
    std::string output_frame_;
    std::string sensor_info_path_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_publisher_;
    std::vector<rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr> image_publishers_;
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
