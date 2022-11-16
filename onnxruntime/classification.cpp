#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#include <tensorrt_provider_factory.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <vector>

std::vector<std::string> read_class_name_list()
{
    std::vector<std::string> class_names;
    std::ifstream ifs("./data/classification_classes_ILSVRC2012.txt");
    std::string line;
    while (getline(ifs, line)) {
        std::size_t pos = line.find(",");
        if (pos != std::string::npos) {
            class_names.push_back(line.substr(0, pos));
        } else {
            class_names.push_back(line);
        }
    }
    ifs.close();
    return class_names;
}

cv::Mat get_inputs(const char* p_str_img_path, int resize_size = 256,
                   int crop_size = 224,
                   cv::Scalar mean = cv::Scalar(0.485, 0.456, 0.406),
                   cv::Scalar std = cv::Scalar(0.229, 0.224, 0.225))
{
    cv::Size inter_size(resize_size, resize_size);
    cv::Size dst_size(crop_size, crop_size);
    cv::Mat image = cv::imread(p_str_img_path);
    cv::resize(image, image, inter_size);
    int start = (resize_size - crop_size) / 2;
    image = image(cv::Rect(start, start, crop_size, crop_size));

    cv::Mat blob;
    mean *= 255;
    cv::dnn::blobFromImage(image, blob, 1.0 / 255, dst_size, mean, true, false,
                           CV_32F);
    cv::divide(blob, std, blob);
    return blob;
}

void softmax(std::vector<float>& values)
{
    float sum = 0;
    float max_value = *std::max_element(values.begin(), values.end());
    for (size_t i = 0; i < values.size(); i++) {
        values[i] = exp(values[i] - max_value);
        sum += values[i];
    }
    for (size_t i = 0; i < values.size(); i++) {
        values[i] /= sum;
    }
}

std::wstring to_wstring(std::string const& str)
{
    size_t len = mbstowcs(nullptr, &str[0], 0);
    std::wstring wstr(len, 0);
    mbstowcs(&wstr[0], &str[0], wstr.size());
    return wstr;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << argv[0] << " provider" << std::endl;
        return -1;
    }
    std::string str_provider(argv[1]);

    std::vector<std::string> class_names = read_class_name_list();

    const char* p_str_img_path = "data/tiger.jpg";
    cv::Mat image = get_inputs(p_str_img_path);

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnxruntime");
    const char* p_str_onnx_path = "data/resnet101.onnx";
    Ort::SessionOptions session_options;

    int device_id = 0;
    if (str_provider == "cuda") {
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(
            session_options, device_id));
    }
    if (str_provider == "tensorrt") {
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(
            session_options, device_id));
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(
            session_options, device_id));
    }

    #ifdef WIN32
        std::wstring wstr_onnx_path = to_wstring(std::string(p_str_onnx_path));
        Ort::Session session(env, wstr_onnx_path.c_str(), session_options);
    #else
        Ort::Session session(env, p_str_onnx_path, session_options);
    #endif

    std::vector<int64_t> input_node_dims = {1, 3, 224, 224};
    constexpr size_t input_tensor_size = 224 * 224 * 3;
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, (float*)image.data, input_tensor_size,
        input_node_dims.data(), 4);

    std::vector<const char*> input_node_names = {"input"};
    std::vector<const char*> output_node_names = {"predict"};

    std::vector<Ort::Value> output_tensors =
        session.Run(Ort::RunOptions{nullptr}, input_node_names.data(),
                    &input_tensor, 1, output_node_names.data(), 1);

    int count = 200;
    std::chrono::system_clock::time_point start =
        std::chrono::system_clock::now();
    for (int i = 0; i < count; i++) {
        output_tensors =
            session.Run(Ort::RunOptions{nullptr}, input_node_names.data(),
                        &input_tensor, 1, output_node_names.data(), 1);
    }
    std::chrono::system_clock::time_point end =
        std::chrono::system_clock::now();
    std::chrono::milliseconds duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double time_cost = duration.count() * 1000.0 / count *
                       std::chrono::milliseconds::period::num /
                       std::chrono::milliseconds::period::den;
    std::cout << "average predict time: " << time_cost << " ms" << std::endl;

    Ort::Value& output_tensor = output_tensors[0];

    Ort::TensorTypeAndShapeInfo tensor_info =
        output_tensor.GetTensorTypeAndShapeInfo();
    std::vector<float> predicts;
    std::vector<int64_t> shape_info = tensor_info.GetShape();
    int element_count = 1;
    for (size_t i = 0; i < shape_info.size(); i++) {
        element_count *= int(shape_info[i]);
    }
    predicts.resize(element_count);
    std::copy(output_tensor.GetTensorData<float>(),
              output_tensor.GetTensorData<float>() + element_count,
              predicts.begin());
    softmax(predicts);

    std::vector<size_t> indices(predicts.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&predicts](int left, int right) -> bool {
                  return predicts[left] > predicts[right];
              });

    std::cout << "category: " << class_names[indices[0]] << ", "
              << "score: " << predicts[indices[0]] << std::endl;

    return 0;
}
