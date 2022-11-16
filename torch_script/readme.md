# TorchScript C++调用说明

---

### 1. 模型序列化

main.py，使用python api，将PyTorch ResNet50模型编译成TorchScript格式，并序列化保存到本地。

### 2. C++ 调用

main.cpp，使用LibTorch库，基于C++接口，实现ResNet50模型的前向推导。
