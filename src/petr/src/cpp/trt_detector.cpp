#include "opencv2/core/types.hpp"
#include "opencv2/imgproc.hpp"
#include "trt_model.hpp"
#include "utils.hpp" 
#include "trt_logger.hpp"

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <algorithm>
#include <cstdint>
#include <string>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc//imgproc.hpp"
#include "opencv2/opencv.hpp"
#include "trt_detector.hpp"
#include "trt_preprocess.hpp"
#include "coco_labels.hpp"

using namespace std;
using namespace nvinfer1;

namespace model{

namespace detector {

namespace {

std::size_t get_element_count(const nvinfer1::Dims& dims) {
    std::size_t count = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        count *= static_cast<std::size_t>(dims.d[i]);
    }
    return count;
}

std::size_t get_data_type_size(nvinfer1::DataType data_type) {
    switch (data_type) {
        case nvinfer1::DataType::kFLOAT:
            return sizeof(float);
        case nvinfer1::DataType::kHALF:
            return sizeof(std::uint16_t);
        case nvinfer1::DataType::kINT32:
            return sizeof(std::int32_t);
        case nvinfer1::DataType::kBOOL:
            return sizeof(bool);
        default:
            return 0;
    }
}

const char* get_data_type_name(nvinfer1::DataType data_type) {
    switch (data_type) {
        case nvinfer1::DataType::kFLOAT:
            return "float32";
        case nvinfer1::DataType::kHALF:
            return "float16";
        case nvinfer1::DataType::kINT32:
            return "int32";
        case nvinfer1::DataType::kBOOL:
            return "bool";
        default:
            return "unknown";
    }
}

}  // namespace

float iou_calc(bbox bbox1, bbox bbox2){
    auto inter_x0 = std::max(bbox1.x0, bbox2.x0);
    auto inter_y0 = std::max(bbox1.y0, bbox2.y0);
    auto inter_x1 = std::min(bbox1.x1, bbox2.x1);
    auto inter_y1 = std::min(bbox1.y1, bbox2.y1);

    float inter_w = inter_x1 - inter_x0;
    float inter_h = inter_y1 - inter_y0;
    
    float inter_area = inter_w * inter_h;
    float union_area = 
        (bbox1.x1 - bbox1.x0) * (bbox1.y1 - bbox1.y0) + 
        (bbox2.x1 - bbox2.x0) * (bbox2.y1 - bbox2.y0) - 
        inter_area;
    
    return inter_area / union_area;
}


void Detector::setup(void const* data, size_t size) {
   /*
     * detector setup需要做的事情
     *   创建engine, context
     *   设置bindings。这里需要注意，不同版本的yolo的输出binding可能还不一样
     *   分配memory空间。这里需要注意，不同版本的yolo的输出所需要的空间也还不一样
     */

    m_runtime     = shared_ptr<IRuntime>(createInferRuntime(*m_logger), destroy_trt_ptr<IRuntime>);
    m_engine      = shared_ptr<ICudaEngine>(m_runtime->deserializeCudaEngine(data, size), destroy_trt_ptr<ICudaEngine>);
    m_context     = shared_ptr<IExecutionContext>(m_engine->createExecutionContext(), destroy_trt_ptr<IExecutionContext>);
    
    int numBindings = m_engine->getNbBindings();
    assert(numBindings == 4); // 1 input, 3 outputs for PETR

    m_inputDims = m_context->getBindingDimensions(0);
    m_outputDims_scores = m_context->getBindingDimensions(1);
    m_outputDims_labels = m_context->getBindingDimensions(2);
    m_outputDims_bboxes = m_context->getBindingDimensions(3);
    m_outputType_scores = m_engine->getBindingDataType(1);
    m_outputType_labels = m_engine->getBindingDataType(2);
    m_outputType_bboxes = m_engine->getBindingDataType(3);

    CUDA_CHECK(cudaStreamCreate(&m_stream));
    
    m_inputSize     = m_params->n_view * m_params->img.h * m_params->img.w * m_params->img.c * sizeof(float);
    m_imgArea       = m_params->img.h * m_params->img.w;
    m_outputSize_scores = get_element_count(m_outputDims_scores) * get_data_type_size(m_outputType_scores);
    m_outputSize_labels = get_element_count(m_outputDims_labels) * get_data_type_size(m_outputType_labels);
    m_outputSize_bboxes = get_element_count(m_outputDims_bboxes) * get_data_type_size(m_outputType_bboxes);

    if (m_outputSize_scores == 0 || m_outputSize_labels == 0 || m_outputSize_bboxes == 0) {
        LOGE("Unsupported output data type: scores=%d, labels=%d, bboxes=%d",
             static_cast<int>(m_outputType_scores),
             static_cast<int>(m_outputType_labels),
             static_cast<int>(m_outputType_bboxes));
        return;
    }

    // 这里对host和device上的memory一起分配空间
    CUDA_CHECK(cudaMallocHost(&m_inputMemory[0], m_inputSize));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&m_outputMemory_scores[0]), m_outputSize_scores));
    CUDA_CHECK(cudaMallocHost(&m_outputMemory_labels[0], m_outputSize_labels));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&m_outputMemory_bboxes[0]), m_outputSize_bboxes));
    CUDA_CHECK(cudaMalloc(&m_inputMemory[1], m_inputSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_outputMemory_scores[1]), m_outputSize_scores));
    CUDA_CHECK(cudaMalloc(&m_outputMemory_labels[1], m_outputSize_labels));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_outputMemory_bboxes[1]), m_outputSize_bboxes));

    // 创建m_bindings
    m_bindings[0] = m_inputMemory[1];
    m_bindings[1] = m_outputMemory_scores[1];
    m_bindings[2] = m_outputMemory_labels[1];
    m_bindings[3] = m_outputMemory_bboxes[1];

    LOGV("TensorRT bindings for PETR:");
    for (int i = 0; i < numBindings; ++i) {
        const char* binding_name = m_engine->getBindingName(i);
        const bool is_input = m_engine->bindingIsInput(i);
        const nvinfer1::Dims binding_dims = m_context->getBindingDimensions(i);
        const nvinfer1::DataType binding_type = m_engine->getBindingDataType(i);
        LOGV("  binding[%d]: name=%s, role=%s, dtype=%s, dims=%s",
             i,
             binding_name != nullptr ? binding_name : "<null>",
             is_input ? "input" : "output",
             get_data_type_name(binding_type),
             printDims(binding_dims).c_str());
    }
}

void Detector::reset_task(){
    m_bboxes.clear();
    m_bboxes3d.clear();
}

bool Detector::preprocess_cpu(cv::Mat image) {
    return true;
}

bool Detector::preprocess_gpu(cv::Mat image) {
    return true;
}

bool Detector::preprocess_cpu_multi(const std::vector<cv::Mat>& views) {
    /*Preprocess -- PETR的多视图预处理 */
    constexpr float kMeanB = 103.530f;
    constexpr float kMeanG = 116.280f;
    constexpr float kMeanR = 123.675f;

    if (views.size() != m_params->n_view) {
        LOGE("ERROR: Number of views %d does not match n_view %d", views.size(), m_params->n_view);
        return false;
    }

    /*Preprocess -- 测速*/
    m_timer->start_cpu();

    int view_size = m_params->img.h * m_params->img.w * m_params->img.c;
    float* input_ptr = m_inputMemory[0];

    for (int v = 0; v < m_params->n_view; v++) {
        cv::Mat image = views[v];
        if (image.data == nullptr) {
            LOGE("ERROR: Image file not founded for view %d!", v);
            return false;
        }

        /*Preprocess -- resize*/
        cv::resize(image, image, 
                   cv::Size(m_params->img.w, m_params->img.h), 0, 0, cv::INTER_LINEAR);

        /*Preprocess -- host端进行BGR减均值归一化, NHWC->NCHW*/
        int index;
        int offset_ch0 = 0;             // B
        int offset_ch1 = m_imgArea;     // G
        int offset_ch2 = m_imgArea * 2; // R
        for (int i = 0; i < m_params->img.h; i++) {
            for (int j = 0; j < m_params->img.w; j++) {
                index = i * m_params->img.w * m_params->img.c + j * m_params->img.c;
                input_ptr[offset_ch0++] = static_cast<float>(image.data[index + 0]) - kMeanB;
                input_ptr[offset_ch1++] = static_cast<float>(image.data[index + 1]) - kMeanG;
                input_ptr[offset_ch2++] = static_cast<float>(image.data[index + 2]) - kMeanR;
            }
        }
        input_ptr += view_size;
    }

    /*Preprocess -- 将host的数据移动到device上*/
    CUDA_CHECK(cudaMemcpyAsync(m_inputMemory[1], m_inputMemory[0], m_inputSize, cudaMemcpyKind::cudaMemcpyHostToDevice, m_stream));

    m_timer->stop_cpu();
    m_timer->duration_cpu<timer::Timer::ms>("preprocess(CPU)");
    return true;
}

bool Detector::preprocess_gpu_multi(const std::vector<cv::Mat>& views) {
    /*Preprocess -- PETR的多视图预处理，使用GPU */

    if (views.size() != m_params->n_view) {
        LOGE("ERROR: Number of views %d does not match n_view %d", views.size(), m_params->n_view);
        return false;
    }

    /*Preprocess -- 测速*/
    m_timer->start_gpu();

    int view_size = m_params->img.h * m_params->img.w * m_params->img.c * sizeof(float);
    float* input_ptr = m_inputMemory[1];

    for (int v = 0; v < m_params->n_view; v++) {
        cv::Mat image = views[v];
        if (image.data == nullptr) {
            LOGE("ERROR: Image file not founded for view %d!", v);
            return false;
        }

        /*Preprocess -- 使用GPU进行warpAffine, 并将结果返回到input_ptr中*/
        preprocess::preprocess_resize_gpu(image, input_ptr,
                                       m_params->img.h, m_params->img.w, 
                                       preprocess::tactics::GPU_WARP_AFFINE);
        input_ptr += m_params->img.h * m_params->img.w * m_params->img.c;
    }

    m_timer->stop_gpu();
    m_timer->duration_gpu("preprocess(GPU)");
    return true;
}


bool Detector::postprocess_cpu() {
    m_timer->start_cpu();
    constexpr float kScoreThreshold = 0.8f;
    constexpr std::size_t kScoreSampleCount = 16;

    /*Postprocess -- 将device上的数据移动到host上*/
    CUDA_CHECK(cudaMemcpyAsync(m_outputMemory_scores[0], m_outputMemory_scores[1], m_outputSize_scores, cudaMemcpyKind::cudaMemcpyDeviceToHost, m_stream));
    CUDA_CHECK(cudaMemcpyAsync(m_outputMemory_labels[0], m_outputMemory_labels[1], m_outputSize_labels, cudaMemcpyKind::cudaMemcpyDeviceToHost, m_stream));
    CUDA_CHECK(cudaMemcpyAsync(m_outputMemory_bboxes[0], m_outputMemory_bboxes[1], m_outputSize_bboxes, cudaMemcpyKind::cudaMemcpyDeviceToHost, m_stream));
    CUDA_CHECK(cudaStreamSynchronize(m_stream));

    if (m_outputType_scores != nvinfer1::DataType::kFLOAT ||
        m_outputType_bboxes != nvinfer1::DataType::kFLOAT) {
        LOGE("Unsupported output dtype for PETR decode: scores=%d, bboxes=%d",
             static_cast<int>(m_outputType_scores),
             static_cast<int>(m_outputType_bboxes));
        return false;
    }

    const std::size_t score_count = get_element_count(m_outputDims_scores);
    const std::size_t label_count = get_element_count(m_outputDims_labels);
    const std::size_t bbox_count = get_element_count(m_outputDims_bboxes);
    LOG("Output tensor shapes: scores=%s, labels=%s, bboxes=%s",
        printDims(m_outputDims_scores).c_str(),
        printDims(m_outputDims_labels).c_str(),
        printDims(m_outputDims_bboxes).c_str());

    const int bbox_dim = (m_outputDims_bboxes.nbDims > 0)
        ? m_outputDims_bboxes.d[m_outputDims_bboxes.nbDims - 1]
        : 0;

    if (bbox_dim != 10) {
        LOGE("Unexpected bbox dim: %d, expected 10", bbox_dim);
        return false;
    }

    const std::size_t num_preds = bbox_count / 10;
    if (score_count != num_preds || label_count != num_preds) {
        LOGE("Mismatched output sizes: scores=%zu, labels=%zu, bboxes=%zu (num_preds=%zu)",
             score_count, label_count, bbox_count, num_preds);
        return false;
    }

    const float* scores = m_outputMemory_scores[0];
    const float* bboxes = m_outputMemory_bboxes[0];

    if (num_preds > 0) {
        float min_score = scores[0];
        float max_score = scores[0];
        double sum_score = 0.0;
        for (std::size_t i = 0; i < num_preds; ++i) {
            min_score = std::min(min_score, scores[i]);
            max_score = std::max(max_score, scores[i]);
            sum_score += scores[i];
        }

        LOG("decoded_scores stats: min=%.6f, max=%.6f, mean=%.6f",
            min_score, max_score, static_cast<float>(sum_score / static_cast<double>(num_preds)));

        const std::size_t sample_count = std::min(num_preds, kScoreSampleCount);
        for (std::size_t i = 0; i < sample_count; ++i) {
            LOG("decoded_scores[%zu] = %.6f", i, scores[i]);
        }
    }

    for (std::size_t i = 0; i < num_preds; ++i) {
        int label = 0;
        if (m_outputType_labels == nvinfer1::DataType::kINT32) {
            label = static_cast<const std::int32_t*>(m_outputMemory_labels[0])[i];
        } else if (m_outputType_labels == nvinfer1::DataType::kFLOAT) {
            label = static_cast<int>(static_cast<const float*>(m_outputMemory_labels[0])[i]);
        } else {
            LOGE("Unsupported label dtype for PETR decode: %d", static_cast<int>(m_outputType_labels));
            return false;
        }

        const float confidence = scores[i];
        if (confidence < kScoreThreshold) {
            continue;
        }
        const float* bbox_ptr = bboxes + i * 10;

        bbox3d box(
            bbox_ptr[0], bbox_ptr[1], bbox_ptr[2], bbox_ptr[3], bbox_ptr[4],
            bbox_ptr[5], bbox_ptr[6], bbox_ptr[7], bbox_ptr[8], bbox_ptr[9],
            confidence, label);
        m_bboxes3d.emplace_back(box);
    }

    LOGV("the count of decoded 3D bbox is %d", m_bboxes3d.size());

    LOGD("\tResult:");
    for (auto& box : m_bboxes3d) {
        LOGD("3D BBox: cx=%.2f, cy=%.2f, cz=%.2f, w=%.2f, l=%.2f, h=%.2f, rot_sine=%.2f, rot_cosine=%.2f, vx=%.2f, vy=%.2f, conf=%.2f, label=%d",
            box.cx, box.cy, box.cz, box.w, box.l, box.h, box.rot_sine, box.rot_cosine, box.vx, box.vy, box.confidence, box.label);
    }
    LOGD("\t\tDetected 3D Objects: %d", m_bboxes3d.size());
    LOGD("");

    m_timer->stop_cpu();
    m_timer->duration_cpu<timer::Timer::ms>("postprocess(CPU)");

    return true;
}


bool Detector::postprocess_gpu() {
    return postprocess_cpu();
}

void Detector::inference_multi(const std::vector<cv::Mat>& views) {
    reset_task();
    // if (m_params->dev == CPU) {
    //     preprocess_cpu_multi(views);
    // } else {
    //     preprocess_gpu_multi(views);
    // }
    preprocess_cpu_multi(views);
    enqueue_bindings();
    postprocess_cpu();
}

const std::vector<bbox3d>& Detector::get_bboxes3d() const {
    return m_bboxes3d;
}

shared_ptr<Detector> make_detector(
    std::string onnx_path, logger::Level level, Params params)
{
    return make_shared<Detector>(onnx_path, level, params);
}

}; // namespace detector
}; // namespace model
