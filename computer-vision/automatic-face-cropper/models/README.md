# YuNet face detection model

This directory contains `face_detection_yunet_2023mar.onnx`, the official YuNet face detection model from the [OpenCV Zoo face detection directory](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).

The project loads the model through OpenCV's `FaceDetectorYN` API. The `2023mar` file is used because this project targets OpenCV 4.x; OpenCV Zoo documents the newer `2026may` export for the OpenCV 5.x ONNX Runtime engine.

All files in the official OpenCV Zoo YuNet directory are distributed under the [MIT License](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/LICENSE).

This model detects face locations. It is not used for face recognition, identity verification, embeddings, or comparison of people.
