import cv2
import torch
import numpy as np
import csv
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from ultralytics import YOLO
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import os


yolo = YOLO(r"D:\PROJECTS\Project1\best (1).pt")
resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
resnet.fc = torch.nn.Identity()
resnet.eval()


preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def test_yolo_classes(video_path, n_frames=5):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    print(f"\n Testing YOLO on {video_path}")
    while cap.isOpened() and frame_count < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        results = yolo(frame)
        if results and results[0].boxes:
            print(f"Frame {frame_count}:")
            for box in results[0].boxes:
                print(f" → Class ID: {int(box.cls)}, Confidence: {float(box.conf)}")
        else:
            print(f"Frame {frame_count}: No detections")
        frame_count += 1
    cap.release()


def extract_features(video_path, player_class=0, max_frames=100):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    features = {}
    bboxes = {}

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo(frame)
        if not results or not results[0].boxes:
            frame_count += 1
            continue

        for i, det in enumerate(results[0].boxes):
            cls = int(det.cls)
            if cls != player_class:
                continue

            x1, y1, x2, y2 = map(int, det.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            player_crop = frame[y1:y2, x1:x2]

            if player_crop.shape[0] < 10 or player_crop.shape[1] < 10:
                continue

            try:
                input_tensor = preprocess(player_crop).unsqueeze(0)
                with torch.no_grad():
                    feat = resnet(input_tensor).squeeze().numpy()
                key = (frame_count, i)
                features[key] = feat
                bboxes[key] = (x1, y1, x2, y2)
            except Exception as e:
                print(f"Error in frame {frame_count}, box {i}: {e}")
        frame_count += 1
    cap.release()
    return features, bboxes


def match_players(t_feats, b_feats):
    t_keys = list(t_feats.keys())
    b_keys = list(b_feats.keys())

    t_vecs = np.array([t_feats[k] for k in t_keys])
    b_vecs = np.array([b_feats[k] for k in b_keys])

    dist_matrix = cdist(t_vecs, b_vecs, metric="cosine")

    row_ind, col_ind = linear_sum_assignment(dist_matrix)

    matches = {}
    for t_idx, b_idx in zip(row_ind, col_ind):
        matches[t_keys[t_idx]] = b_keys[b_idx]
    return matches


def save_matches_to_csv(matches, path="matches.csv"):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Tacticam_FrameID", "Tacticam_PlayerBox", "Broadcast_FrameID", "Broadcast_PlayerBox"])
        for t_id, b_id in matches.items():
            writer.writerow([t_id[0], t_id[1], b_id[0], b_id[1]])
    print(f" Player ID match results saved to {path}")


def save_annotated_frames(video_path, bboxes, ids, output_dir="annotated_frames", label_prefix=""):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_index = 0
    saved_ids = set(k[0] for k in ids)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        for (f_idx, box_id), bbox in bboxes.items():
            if f_idx == frame_index:
                x1, y1, x2, y2 = bbox
                label = f"{label_prefix}{box_id}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if frame_index in saved_ids:
            cv2.imwrite(f"{output_dir}/frame_{frame_index}.jpg", frame)
        frame_index += 1
    cap.release()


if __name__ == "__main__":
    

    player_class_id = 0  

    print(" Extracting features from tacticam")
    tacticam_feats, tacticam_boxes = extract_features(r"D:\PROJECTS\Project1\tacticam.mp4", player_class=player_class_id)

    print(" Extracting features from broadcast")
    broadcast_feats, broadcast_boxes = extract_features(r"D:\PROJECTS\Project1\broadcast.mp4", player_class=player_class_id)

    print("Matching players")
    matches = match_players(tacticam_feats, broadcast_feats)

    print(" Saving match results to CSV")
    save_matches_to_csv(matches, "player_matches.csv")

    
    print(" Saving sample annotated frames")
    save_annotated_frames(r"D:\PROJECTS\Project1\tacticam.mp4", tacticam_boxes, matches.keys(), output_dir="tacticam_vis", label_prefix="T")
    save_annotated_frames(r"D:\PROJECTS\Project1\broadcast.mp4", broadcast_boxes, matches.values(), output_dir="broadcast_vis", label_prefix="B")

    
