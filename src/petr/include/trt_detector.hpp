#ifndef __TRT_DETECTOR_HPP__
#define __TRT_DETECTOR_HPP__

#include <memory>
#include <vector>
#include <string>
#include "NvInfer.h"
#include "trt_logger.hpp"
#include "trt_model.hpp"

namespace model{

namespace detector {

enum model {
    YOLOV5,
    YOLOV8
};

struct bbox {
    float x0, x1, y0, y1;
    float confidence;
    bool  flg_remove;
    int   label;
    
    bbox() = default;
    bbox(float x0, float y0, float x1, float y1, float conf, int label) : 
        x0(x0), y0(y0), x1(x1), y1(y1), 
        confidence(conf), flg_remove(false), 
        label(label){};
};

struct bbox3d {
    float cx, cy, cz, w, l, h, rot_sine, rot_cosine, vx, vy;
    float confidence;
    bool  flg_remove;
    int   label;
    
    bbox3d() = default;
    bbox3d(float cx, float cy, float cz, float w, float l, float h, float rot_sine, float rot_cosine, float vx, float vy, float conf, int label) : 
        cx(cx), cy(cy), cz(cz), w(w), l(l), h(h), rot_sine(rot_sine), rot_cosine(rot_cosine), vx(vx), vy(vy),
        confidence(conf), flg_remove(false), 
        label(label){};
};

class Detector : public Model{

public:
    // 这个构造函数实际上调用的是父类的Model的构造函数
    Detector(std::string onnx_path, logger::Level level, Params params) : 
        Model(onnx_path, level, params) {};

public:
    // 这里detection自己实现了一套前处理/后处理，以及内存分配的初始化
    virtual void setup(void const* data, std::size_t size) override;
    virtual void reset_task() override;
    virtual bool preprocess_cpu(cv::Mat image) override;
    virtual bool preprocess_gpu(cv::Mat image) override;
    virtual bool postprocess_cpu() override;
    virtual bool postprocess_gpu() override;

    // For PETR multi-view
    bool preprocess_cpu_multi(const std::vector<cv::Mat>& views);
    bool preprocess_gpu_multi(const std::vector<cv::Mat>& views);
    void inference_multi(const std::vector<cv::Mat>& views);
    const std::vector<bbox3d>& get_bboxes3d() const;

private:
    std::vector<bbox> m_bboxes;
    std::vector<bbox3d> m_bboxes3d;
    int m_inputSize; 
    int m_imgArea;
    // For PETR: scores, labels, bboxes
    float* m_outputMemory_scores[2];
    void* m_outputMemory_labels[2];
    float* m_outputMemory_bboxes[2];
    nvinfer1::Dims m_outputDims_scores;
    nvinfer1::Dims m_outputDims_labels;
    nvinfer1::Dims m_outputDims_bboxes;
    nvinfer1::DataType m_outputType_scores;
    nvinfer1::DataType m_outputType_labels;
    nvinfer1::DataType m_outputType_bboxes;
    std::size_t m_outputSize_scores;
    std::size_t m_outputSize_labels;
    std::size_t m_outputSize_bboxes;
};

// 外部调用的接口
std::shared_ptr<Detector> make_detector(
    std::string onnx_path, logger::Level level, Params params);

}; // namespace detector
}; // namespace model

#endif //__TRT_DETECTOR_HPP__
