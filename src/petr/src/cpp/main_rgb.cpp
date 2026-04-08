#include "trt_logger.hpp"
#include "trt_model.hpp"
#include "trt_worker.hpp"
#include "utils.hpp"

#include <memory>
#include <string>
#include <vector>

// ROS2 & CV Includes
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>

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

        // 创建一个worker的实例, 在创建的时候就完成初始化
        worker = thread::create_worker(onnxPath, level, params);

        current_images.resize(6);

        /* senser_msgs::msg::Image 是原始 RGB 图像 */
        std::string topics[6] = {"/FRONT_CAMERA",
                                 "/FRONT_LEFT_CAMERA",
                                 "/FRONT_RIGHT_CAMERA",
                                 "/BACK_CAMERA",
                                 "/BACK_LEFT_CAMERA",
                                 "/BACK_RIGHT_CAMERA"};

        /* sensor_msgs::msg::CompressedImage 是压缩后的图像，需要解码 */
        // std::string topics[6] = {"/FRONT_CAMERA/compressed",
        //                          "/FRONT_LEFT_CAMERA/compressed",
        //                          "/FRONT_RIGHT_CAMERA/compressed",
        //                          "/BACK_CAMERA/compressed",
        //                          "/BACK_LEFT_CAMERA/compressed",
        //                          "/BACK_RIGHT_CAMERA/compressed"};

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

    void syncedCallback(const sensor_msgs::msg::Image::ConstSharedPtr &img0,
                        const sensor_msgs::msg::Image::ConstSharedPtr &img1,
                        const sensor_msgs::msg::Image::ConstSharedPtr &img2,
                        const sensor_msgs::msg::Image::ConstSharedPtr &img3,
                        const sensor_msgs::msg::Image::ConstSharedPtr &img4,
                        const sensor_msgs::msg::Image::ConstSharedPtr &img5)
    {
        // Convert all images to cv::Mat and forward to the worker
        std::vector<cv::Mat> imgs(6);
        // 使用 toCvShare 避免深拷贝，除非你需要修改图像内容（infer通常只需要读取）
        // 如果 worker 内部会修改图像，则保持用 toCvCopy
        imgs[0] = cv_bridge::toCvCopy(img0, sensor_msgs::image_encodings::BGR8)->image;
        imgs[1] = cv_bridge::toCvCopy(img1, sensor_msgs::image_encodings::BGR8)->image;
        imgs[2] = cv_bridge::toCvCopy(img2, sensor_msgs::image_encodings::BGR8)->image;
        imgs[3] = cv_bridge::toCvCopy(img3, sensor_msgs::image_encodings::BGR8)->image;
        imgs[4] = cv_bridge::toCvCopy(img4, sensor_msgs::image_encodings::BGR8)->image;
        imgs[5] = cv_bridge::toCvCopy(img5, sensor_msgs::image_encodings::BGR8)->image;

        worker->inference_multi(imgs);
    }

private:
    // message_filters subscribers for each camera
    message_filters::Subscriber<sensor_msgs::msg::Image> sub_cam0_;
    message_filters::Subscriber<sensor_msgs::msg::Image> sub_cam1_;
    message_filters::Subscriber<sensor_msgs::msg::Image> sub_cam2_;
    message_filters::Subscriber<sensor_msgs::msg::Image> sub_cam3_;
    message_filters::Subscriber<sensor_msgs::msg::Image> sub_cam4_;
    message_filters::Subscriber<sensor_msgs::msg::Image> sub_cam5_;

    using MySyncPolicy = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::Image,
        sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::Image>;

    std::shared_ptr<message_filters::Synchronizer<MySyncPolicy>> sync_;

    std::vector<cv::Mat> current_images;
    std::shared_ptr<thread::Worker> worker;
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
