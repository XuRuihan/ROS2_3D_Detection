#include "trt_worker.hpp"
#include "trt_classifier.hpp"
#include "trt_detector.hpp"
#include "trt_logger.hpp"
#include "memory"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <sstream>

using namespace std;

namespace thread{

Worker::Worker(string onnxPath, logger::Level level, model::Params params) {
    m_logger = logger::create_logger(level);

    // 这里根据task_type选择创建的trt_model的子类，今后会针对detection, segmentation扩充
    if (params.task == model::task_type::CLASSIFICATION) 
        m_classifier = model::classifier::make_classifier(onnxPath, level, params);
    else if (params.task == model::task_type::DETECTION) 
        m_detector = model::detector::make_detector(onnxPath, level, params);
}

void Worker::inference(cv::Mat image) {
    auto now = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch());

    stringstream ss;
    ss << "data/source/scene_" << duration.count() << ".jpg";
    string imagePath = ss.str();
    if (m_classifier != nullptr) {
        m_classifier->init_model();
        // m_classifier->load_image(imagePath);
        m_classifier->m_imagePath = imagePath;
        m_classifier->inference(image);
    }

    if (m_detector != nullptr) {
        m_detector->init_model();
        // m_detector->load_image(imagePath);
        m_detector->m_imagePath = imagePath;
        m_detector->inference(image);
    }
}

void Worker::inference_multi(const std::vector<cv::Mat>& views) {
    if (m_detector != nullptr) {
        m_detector->init_model();
        m_detector->inference_multi(views);
    }
}

std::vector<model::detector::bbox3d> Worker::inference_multi_and_get_result(const std::vector<cv::Mat>& views) {
    if (m_detector == nullptr) {
        return {};
    }

    m_detector->init_model();
    m_detector->inference_multi(views);
    return m_detector->get_bboxes3d();
}

shared_ptr<Worker> create_worker(
    std::string onnxPath, logger::Level level, model::Params params) 
{
    // 使用智能指针来创建一个实例
    return make_shared<Worker>(onnxPath, level, params);
}

}; // namespace thread
