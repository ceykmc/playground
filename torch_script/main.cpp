#include <torch/script.h>
#include <torch/torch.h>

#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

torch::Tensor get_input(const char* p_str_img_path)
{
    cv::Mat image = cv::imread(p_str_img_path);
    cv::resize(image, image, cv::Size(256, 256));

    cv::Size crop_size(224, 224);
    int x_offset = (image.cols - crop_size.width) / 2;
    int y_offset = (image.rows - crop_size.height) / 2;
    cv::Rect roi(x_offset, y_offset, crop_size.width, crop_size.height);
    image = image(roi);

    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    image.convertTo(image, CV_32FC3, 1.0 / 255.0);

    torch::Tensor tensor =
        torch::from_blob(image.data, {image.rows, image.cols, 3}, c10::kFloat);
    tensor = tensor.permute({2, 0, 1});
    tensor.unsqueeze_(0);

    std::vector<double> norm_mean = {0.485, 0.456, 0.406};
    std::vector<double> norm_std = {0.229, 0.224, 0.225};
    tensor = torch::data::transforms::Normalize<>(norm_mean, norm_std)(tensor);
    return tensor.clone();
}

int main(int, char**)
{
    if (torch::cuda::is_available()) {
        std::cout << "cuda is available" << std::endl;
    } else {
        std::cout << "cuda is not available" << std::endl;
    }
    if (torch::cuda::cudnn_is_available()) {
        std::cout << "cudnn is available" << std::endl;
    } else {
        std::cout << "cudnn is not available" << std::endl;
    }

    const char* p_str_img_path = "data/dog.jpeg";
    cv::Mat image = cv::imread(p_str_img_path);
    if (image.empty()) {
        std::cerr << "can not load image: " << p_str_img_path << std::endl;
        return 1;
    }
    torch::Tensor input = get_input(p_str_img_path);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);

    const char* p_str_jit_module_path = "data/resnet50.ts";
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(p_str_jit_module_path);
    } catch (const c10::Error& e) {
        std::cerr << "error loading the jit module: " << p_str_jit_module_path
                  << ", error message: " << e.what() << std::endl;
        return 1;
    }

    int warm_up = 20;
    int n_count = 100;
    torch::Tensor prediction;

    // cpu
    for (int i = 0; i < warm_up; i += 1) {
        prediction = module.forward(inputs).toTensor();
    }
    std::chrono::system_clock::time_point start =
        std::chrono::system_clock::now();
    for (int i = 0; i < n_count; i += 1) {
        prediction = module.forward(inputs).toTensor();
    }
    std::chrono::system_clock::time_point end =
        std::chrono::system_clock::now();
    std::chrono::milliseconds duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double time_cost = duration.count() * 1000 *
                       std::chrono::milliseconds::period::num /
                       std::chrono::milliseconds::period::den;
    double avg_time = time_cost / n_count;
    std::cout << "cpu avg time is: " << avg_time << std::endl;

    // cuda
    torch::DeviceType cuda_device = torch::kCUDA;
    std::vector<torch::jit::IValue> cuda_inputs;
    cuda_inputs.push_back(input.to(cuda_device));
    module.to(cuda_device);
    for (int i = 0; i < warm_up; i += 1) {
        prediction = module.forward(cuda_inputs).toTensor();
    }
    start = std::chrono::system_clock::now();
    for (int i = 0; i < n_count; i += 1) {
        prediction = module.forward(cuda_inputs).toTensor();
    }
    end = std::chrono::system_clock::now();
    duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    time_cost = duration.count() * 1000 *
                std::chrono::milliseconds::period::num /
                std::chrono::milliseconds::period::den;
    avg_time = time_cost / n_count;
    std::cout << "gpu avg time is: " << avg_time << std::endl;

    prediction = prediction.to(torch::kCPU);
    torch::Tensor output = torch::softmax(prediction, 1);
    std::tuple<torch::Tensor, torch::Tensor> result = torch::max(output, 1);
    auto score = std::get<0>(result).item();
    auto index = std::get<1>(result).item();

    std::cout << "index is: " << index << std::endl;
    std::cout << "score is: " << score << std::endl;

    return 0;
}
