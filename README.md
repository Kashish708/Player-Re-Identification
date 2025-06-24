# Player-Re-Identification

Match players across different camera views using deep learning!
This project uses YOLO for object detection and ResNet50 for feature extraction to identify and match the same players in two different videos: tactical and Broadcast.

---
## **How It Works**

- It uses Ultralytics YOLO to detect player bounding boxes in each video frame.

- Crop each detected player and pass it through ResNet50 to extract deep features.

- Computes feature similarity using cosine distance and matches using the Hungarian Algorithm.

- Saves matches to a CSV and optionally visualises the results on frames.

---
## **Tech Stack**

- YOLO (Ultralytics) for object detection

- ResNet50 (via PyTorch) for feature extraction

- OpenCV for video processing

- SciPy for optimization and distance computation

- Torchvision for pre-trained models
