# Player-Re-Identification

Match players across different camera views using deep learning!
This project uses YOLO for object detection and ResNet50 for feature extraction to identify and match the same players in two different videos: tactical and Broadcast.

---
## **How It Works**

- It uses Ultralytics YOLO to detect player bounding boxes in each video frame.

- Crop each detected player and pass it through ResNet50 to extract deep features.

- Computes feature similarity using cosine distance and matches using the Hungarian Algorithm.

- Saves matches to a CSV and optionally visualizes the results on frames.

---
## **Tech Stack**

- YOLO (Ultralytics) for object detection

- ResNet50 (via PyTorch) for feature extraction

- OpenCV for video processing

- SciPy for optimization and distance computation

- Torchvision for pre-trained models

---
## **Setup** 

### 1. Clone the repository

```bash
git clone https://github.com/Kashish708/Player-Re-Identification
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add Required Files

- Could you place the YOLO model file (best.pt) in the project directory. 

- Place video files:

-  tacticam.mp4

-  broadcast.mp4

### 4.  Run the script

```bash
python app1.py
```
