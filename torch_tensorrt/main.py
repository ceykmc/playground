import time

import torch
import torchvision
import torch_tensorrt


def get_inputs():
    image_path = "data/dog.jpeg"
    image = torchvision.io.read_image(image_path)

    weights = torchvision.models.ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()
    batch = preprocess(image).unsqueeze(0)

    return batch


def print_output(prediction: torch.Tensor):
    weights = torchvision.models.ResNet50_Weights.DEFAULT
    prediction = prediction.cpu()
    prediction = prediction.squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    print(f"{category_name}: {score:.2%}")


def save_model():
    device = torch.device("cuda:0")

    inputs = get_inputs()
    inputs = inputs.to(device)

    weights = torchvision.models.ResNet50_Weights.DEFAULT
    model = torchvision.models.resnet50(weights=weights)
    model.to(device)
    model.eval()

    with torch.inference_mode():
        ref_fp32 = model(inputs)
    print_output(ref_fp32)

    ts_model = torch.jit.script(model)
    ts_prediction = ts_model(inputs)
    print_output(ts_prediction)
    torch.jit.save(ts_model, "data/resnet50.ts")

    trt_model = torch_tensorrt.compile(
        module=model, inputs=[inputs], enabled_precisions={torch.float32}
    )
    trt_prediction = trt_model(inputs)
    print_output(trt_prediction)
    torch.jit.save(trt_model, "data/resnet50_trt_fp32.ts")

    trt_model = torch_tensorrt.compile(
        module=model, inputs=[inputs.half()], enabled_precisions={torch.float16}
    )
    trt_prediction = trt_model(inputs.half())
    print_output(trt_prediction)
    torch.jit.save(trt_model, "data/resnet50_trt_fp16.ts")


def compute_average_time(model, inputs):
    n_warmup = 50
    n_runs = 200

    for _ in range(n_warmup):
        model(inputs)
    print_output(model(inputs))
    start = time.time()
    for _ in range(n_runs):
        model(inputs)
    end = time.time()
    return (end - start) / n_runs * 1000


def benchmark():
    device = torch.device("cuda:0")

    inputs = get_inputs()
    inputs = inputs.to(device)

    weights = torchvision.models.ResNet50_Weights.DEFAULT
    model = torchvision.models.resnet50(weights=weights)
    model.to(device=device)
    model.eval()

    avg_time = compute_average_time(model, inputs)
    print(f"PyTorch inference time: {avg_time:.2f} ms")

    ts_model = torch.jit.load("data/resnet50.ts")
    avg_time = compute_average_time(ts_model, inputs)
    print(f"TorchScript inference time: {avg_time:.2f} ms")

    trt_model_fp32 = torch.jit.load("data/resnet50_trt_fp32.ts")
    avg_time = compute_average_time(trt_model_fp32, inputs)
    print(f"Torch TensorRT FP32 inference time: {avg_time:.2f} ms")

    trt_model_fp16 = torch.jit.load("data/resnet50_trt_fp16.ts")
    avg_time = compute_average_time(trt_model_fp16, inputs.half())
    print(f"Torch TensorRT FP16 inference time: {avg_time:.2f} ms")


def main():
    # save_model()
    benchmark()


if __name__ == "__main__":
    main()
