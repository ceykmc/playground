import time
import cv2
import numpy as np

import onnxruntime as ort

from labels import COCOLabels


def main():
    image_path = "data/person.jpg"
    show_image = cv2.imread(image_path)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose((2, 0, 1)).astype(np.float32)
    image /= 255.0
    image = image[None, :]

    provider_list = [
        # "CPUExecutionProvider",
        "CUDAExecutionProvider"
    ]

    onnx_model_path = "data/yolov7.onnx"
    session = ort.InferenceSession(onnx_model_path, providers=provider_list)
    predicts = session.run(input_feed={"images": image}, output_names=["output"])[0]

    count = 50
    start = time.time()
    for _ in range(count):
        predicts = session.run(input_feed={"images": image}, output_names=["output"])[0]
    end = time.time()
    avg_time = (end - start) / count * 1000
    print(f"average time: {avg_time:.2f} ms")

    for predict in predicts:
        class_id = int(predict[0])
        x1, y1, x2, y2 = [int(e) for e in predict[1:5]]
        cv2.rectangle(show_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            show_image,
            COCOLabels(class_id).name.lower(),
            (x1, y1 - 10),
            2,
            0.5,
            (0, 0, 255),
        )
    cv2.imshow("show", show_image)
    cv2.waitKey()


if __name__ == "__main__":
    main()
