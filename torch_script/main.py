import time

import torch
import torchvision


def main():
    weights = torchvision.models.ResNet50_Weights.DEFAULT
    model = torchvision.models.resnet50(weights=weights)
    model.eval()

    ts_model = torch.jit.script(model)
    torch.jit.save(ts_model, "data/resnet50.ts")

    image_path = "data/dog.jpeg"
    image = torchvision.io.read_image(image_path)
    preprocess = weights.transforms()
    batch = preprocess(image).unsqueeze(0)

    warm_up = 20
    n_count = 100

    for _ in range(warm_up):
        prediction = model(batch)
    start = time.time()
    for _ in range(n_count):
        prediction = model(batch)
    end = time.time()
    avg_time = (end - start) / n_count * 1000
    print(f"cpu PyTorch average time: {avg_time:.2f}ms")

    for _ in range(warm_up):
        prediction = ts_model(batch)
    start = time.time()
    for _ in range(n_count):
        prediction = ts_model(batch)
    end = time.time()
    avg_time = (end - start) / n_count * 1000
    print(f"cpu TorchScript average time: {avg_time:.2f}ms")

    device = torch.device("cuda:0")
    batch = batch.to(device)
    model = model.to(device)
    for _ in range(warm_up):
        prediction = model(batch)
    start = time.time()
    for _ in range(n_count):
        prediction = model(batch)
    end = time.time()
    avg_time = (end - start) / n_count * 1000
    print(f"gpu PyTorch average time: {avg_time:.2f}ms")

    ts_model = ts_model.to(device)
    for _ in range(warm_up):
        prediction = ts_model(batch)
    start = time.time()
    for _ in range(n_count):
        prediction = ts_model(batch)
    end = time.time()
    avg_time = (end - start) / n_count * 1000
    print(f"gpu TorchScript average time: {avg_time:.2f}ms")

    prediction = prediction.cpu()
    prediction = prediction.squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    print(f"class id: {class_id}, score: {score:.4f}")
    print(f"{category_name}: {100 * score:.1f}%")


if __name__ == "__main__":
    main()
