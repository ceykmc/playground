import time

import cv2
import numpy as np
import scipy
import torch
import torchvision

import onnxruntime as ort


def get_inputs(
    image_path,
    resize_size=256,
    crop_size=224,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    image = cv2.imread(image_path)
    image = cv2.resize(image, dsize=(resize_size, resize_size))
    start = (resize_size - crop_size) // 2
    image = image[start : start + crop_size, start : start + crop_size, :]
    image = image[:, :, ::-1]  # from BGR to RGB
    image = image.astype(np.float32) / 255
    image = (image - mean) / std
    image = image.astype(np.float32)
    image = np.transpose(image, (2, 0, 1))
    return image[None, :]


def print_output(prediction: np.ndarray):
    weights = torchvision.models.ResNet101_Weights.DEFAULT
    prediction = prediction[0]
    prediction = scipy.special.softmax(prediction)
    class_id = np.argmax(prediction)
    score = prediction[class_id]
    category_name = weights.meta["categories"][class_id]
    print(f"{category_name}: {score:.2%}")


def export_onnx_model():
    weights = torchvision.models.ResNet101_Weights.DEFAULT
    model = torchvision.models.resnet101(weights=weights)
    model.eval()

    ts_model = torch.jit.script(model)

    dummy_input = torch.rand(10, 3, 224, 224)
    input_names = ["input"]
    output_names = ["predict"]
    torch.onnx.export(
        model=ts_model,
        args=dummy_input,
        f="data/resnet101.onnx",
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={"input": {0: "batch_size"}, "predict": {0: "batch_size"}},
    )


def main():
    # export_onnx_model()

    image_path = "data/tiger.jpg"
    image = get_inputs(image_path)

    provider_list = ["CPUExecutionProvider"]
    # provider_list = ["CUDAExecutionProvider"]
    # provider_list = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    onnx_model_path = "data/resnet101.onnx"
    session = ort.InferenceSession(onnx_model_path, providers=provider_list)

    count = 200
    start = time.time()
    for _ in range(count):
        predict = session.run(input_feed={"input": image}, output_names=["predict"])
    end = time.time()
    avg_time = (end - start) / count * 1000
    print(f"average time: {avg_time:.2f} ms")

    print_output(predict[0])


if __name__ == "__main__":
    main()
