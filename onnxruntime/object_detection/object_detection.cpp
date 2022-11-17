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

cv::Mat get_inputs(const char *p_str_img_path)
{
    cv::Mat image = cv::imread(p_str_img_path);
    cv::dnn::blobFromImage(image, image, 1.0 / 255, cv::Size(), cv::Scalar(),
                           true, false, CV_32F);
    return image;
}

std::wstring to_wstring(std::string const &str)
{
    size_t len = mbstowcs(nullptr, &str[0], 0);
    std::wstring wstr(len, 0);
    mbstowcs(&wstr[0], &str[0], wstr.size());
    return wstr;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        std::cerr << argv[0] << " provider" << std::endl;
        return -1;
    }
    std::string str_provider(argv[1]);

    const char *p_str_img_path = "data/person.jpg";
    cv::Mat image = get_inputs(p_str_img_path);
    cv::Mat show_image = cv::imread(p_str_img_path);

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnxruntime");
    const char *p_str_onnx_path = "data/yolov7.onnx";
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

    std::vector<int64_t> input_node_dims = {1, 3, 640, 640};
    constexpr size_t input_tensor_size = 640 * 640 * 3;
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, (float *)image.data, input_tensor_size,
        input_node_dims.data(), 4);

    std::vector<const char *> input_node_names = {"images"};
    std::vector<const char *> output_node_names = {"output"};

    std::vector<Ort::Value> output_tensors =
        session.Run(Ort::RunOptions{nullptr}, input_node_names.data(),
                    &input_tensor, 1, output_node_names.data(), 1);

    int count = 50;
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

    Ort::Value &output_tensor = output_tensors[0];

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
    
    for (size_t i = 0; i < predicts.size(); i += 7) {
        int x1 = predicts[i + 1];
        int y1 = predicts[i + 2];
        int x2 = predicts[i + 3];
        int y2 = predicts[i + 4];
        cv::rectangle(show_image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 2);
    }
    cv::imshow("show", show_image);
    cv::waitKey();
    return 0;
}
