# import sys
# import os

# # Add yolov7 folder to Python path
# YOLOV7_PATH = os.path.join(os.path.dirname(__file__), "yolov7")
# sys.path.insert(0, YOLOV7_PATH)

# import torch
# import cv2
# import numpy as np
# import os
# from yolov7.utils.datasets import letterbox
# from yolov7.utils.general import non_max_suppression, scale_coords
# from yolov7.utils.plots import plot_one_box

# # Load models once
# model_garbage = torch.load('model_wts/garbage.pt', map_location='cpu')['model'].float().eval()
# model_building = torch.load('model_wts/building.pt', map_location='cpu')['model'].float().eval()

# def detect_objects(image_path):
#     img0 = cv2.imread(image_path)
#     img = letterbox(img0, new_shape=416)[0]
#     img = img[:, :, ::-1].transpose(2, 0, 1)
#     img = np.ascontiguousarray(img)

#     img_tensor = torch.from_numpy(img).float() / 255.0
#     img_tensor = img_tensor.unsqueeze(0)

#     with torch.no_grad():
#         pred_garbage = model_garbage(img_tensor)[0]
#         pred_building = model_building(img_tensor)[0]

#     pred_garbage = non_max_suppression(pred_garbage, conf_thres=0.15, iou_thres=0.45)[0]
#     pred_building = non_max_suppression(pred_building, conf_thres=0.55, iou_thres=0.45)[0]

#     names_garbage = model_garbage.names
#     names_building = model_building.names

#     garbage_count = 0
#     building_count = 0

#     if pred_garbage is not None and len(pred_garbage):
#         pred_garbage[:, :4] = scale_coords(img_tensor.shape[2:], pred_garbage[:, :4], img0.shape).round()
#         for *xyxy, conf, cls in pred_garbage:
#             garbage_count += 1
#             # label = f"{names_garbage[int(cls)]} {conf:.2f}"
#             label = f"{names_garbage[int(cls)]}"
#             plot_one_box(xyxy, img0, label=label, color=(0, 255, 0), line_thickness=2)

#     if pred_building is not None and len(pred_building):
#         pred_building[:, :4] = scale_coords(img_tensor.shape[2:], pred_building[:, :4], img0.shape).round()
#         for *xyxy, conf, cls in pred_building:
#             building_count += 1
#             # label = f"{names_building[int(cls)]} {conf:.2f}"
#             label = f"{names_building[int(cls)]}"
#             plot_one_box(xyxy, img0, label=label, color=(255, 0, 0), line_thickness=2)

#     output_path = image_path.replace(".jpg", "_detected.jpg")
#     cv2.imwrite(output_path, img0)

#     return output_path, garbage_count, building_count

# import sys
# import os
# import requests
# import torch
# import cv2
# import numpy as np

# # Add yolov7 folder to Python path
# YOLOV7_PATH = os.path.join(os.path.dirname(__file__), "yolov7")
# sys.path.insert(0, YOLOV7_PATH)

# from yolov7.utils.datasets import letterbox
# from yolov7.utils.general import non_max_suppression, scale_coords
# from yolov7.utils.plots import plot_one_box

# # Step 1: Download models from Hugging Face if not already present
# def download_models():
#     urls = {
#         "garbage.pt": "https://huggingface.co/EASYTOCODE99/SafetyModels/resolve/main/garbage.pt",
#         "building.pt": "https://huggingface.co/EASYTOCODE99/SafetyModels/resolve/main/building.pt"
#     }
#     os.makedirs("model_wts", exist_ok=True)
#     for filename, url in urls.items():
#         path = os.path.join("model_wts", filename)
#         if not os.path.exists(path):
#             print(f"⬇️ Downloading {filename}...")
#             with requests.get(url, stream=True) as r:
#                 with open(path, "wb") as f:
#                     for chunk in r.iter_content(chunk_size=8192):
#                         if chunk:
#                             f.write(chunk)
#             print(f"✅ Downloaded {filename}")

# download_models()

# # Step 2: Load YOLO models
# model_garbage = torch.load('model_wts/garbage.pt', map_location='cpu')['model'].float().eval()
# model_building = torch.load('model_wts/building.pt', map_location='cpu')['model'].float().eval()

# # Step 3: Object detection function
# def detect_objects(image_path):
#     img0 = cv2.imread(image_path)
#     img = letterbox(img0, new_shape=416)[0]
#     img = img[:, :, ::-1].transpose(2, 0, 1)
#     img = np.ascontiguousarray(img)

#     img_tensor = torch.from_numpy(img).float() / 255.0
#     img_tensor = img_tensor.unsqueeze(0)

#     with torch.no_grad():
#         pred_garbage = model_garbage(img_tensor)[0]
#         pred_building = model_building(img_tensor)[0]

#     pred_garbage = non_max_suppression(pred_garbage, conf_thres=0.15, iou_thres=0.45)[0]
#     pred_building = non_max_suppression(pred_building, conf_thres=0.55, iou_thres=0.45)[0]

#     names_garbage = model_garbage.names
#     names_building = model_building.names

#     garbage_count = 0
#     building_count = 0

#     if pred_garbage is not None and len(pred_garbage):
#         pred_garbage[:, :4] = scale_coords(img_tensor.shape[2:], pred_garbage[:, :4], img0.shape).round()
#         for *xyxy, conf, cls in pred_garbage:
#             garbage_count += 1
#             label = f"{names_garbage[int(cls)]}"
#             plot_one_box(xyxy, img0, label=label, color=(0, 255, 0), line_thickness=2)

#     if pred_building is not None and len(pred_building):
#         pred_building[:, :4] = scale_coords(img_tensor.shape[2:], pred_building[:, :4], img0.shape).round()
#         for *xyxy, conf, cls in pred_building:
#             building_count += 1
#             label = f"{names_building[int(cls)]}"
#             plot_one_box(xyxy, img0, label=label, color=(255, 0, 0), line_thickness=2)

#     output_path = image_path.replace(".jpg", "_detected.jpg")
#     cv2.imwrite(output_path, img0)

#     return output_path, garbage_count, building_count


import sys
import os
import requests
import torch
import cv2
import numpy as np
import streamlit as st

# Add yolov7 folder to Python path
YOLOV7_PATH = os.path.join(os.path.dirname(__file__), "yolov7")
sys.path.insert(0, YOLOV7_PATH)

from yolov7.utils.datasets import letterbox
from yolov7.utils.general import non_max_suppression, scale_coords
from yolov7.utils.plots import plot_one_box

# Step 1: Download models from Hugging Face if not already present
def download_models():
    urls = {
        "garbage.pt": "https://huggingface.co/EASYTOCODE99/SafetyModels/resolve/main/garbage.pt",
        "building.pt": "https://huggingface.co/EASYTOCODE99/SafetyModels/resolve/main/building.pt"
    }
    os.makedirs("model_wts", exist_ok=True)
    for filename, url in urls.items():
        path = os.path.join("model_wts", filename)
        if not os.path.exists(path):
            with requests.get(url, stream=True) as r:
                with open(path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

# Spinner for loading
with st.spinner("🔄 Downloading & loading YOLO models..."):
    download_models()
    model_garbage = torch.load('model_wts/garbage.pt', map_location='cpu')['model'].float().eval()
    model_building = torch.load('model_wts/building.pt', map_location='cpu')['model'].float().eval()

# Step 2: Object detection function
def detect_objects(image_path):
    img0 = cv2.imread(image_path)
    img = letterbox(img0, new_shape=416)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    img_tensor = torch.from_numpy(img).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        pred_garbage = model_garbage(img_tensor)[0]
        pred_building = model_building(img_tensor)[0]

    pred_garbage = non_max_suppression(pred_garbage, conf_thres=0.15, iou_thres=0.45)[0]
    pred_building = non_max_suppression(pred_building, conf_thres=0.55, iou_thres=0.45)[0]

    names_garbage = model_garbage.names
    names_building = model_building.names

    garbage_count = 0
    building_count = 0

    if pred_garbage is not None and len(pred_garbage):
        pred_garbage[:, :4] = scale_coords(img_tensor.shape[2:], pred_garbage[:, :4], img0.shape).round()
        for *xyxy, conf, cls in pred_garbage:
            garbage_count += 1
            label = f"{names_garbage[int(cls)]}"
            plot_one_box(xyxy, img0, label=label, color=(0, 255, 0), line_thickness=2)

    if pred_building is not None and len(pred_building):
        pred_building[:, :4] = scale_coords(img_tensor.shape[2:], pred_building[:, :4], img0.shape).round()
        for *xyxy, conf, cls in pred_building:
            building_count += 1
            label = f"{names_building[int(cls)]}"
            plot_one_box(xyxy, img0, label=label, color=(255, 0, 0), line_thickness=2)

    output_path = image_path.replace(".jpg", "_detected.jpg")
    cv2.imwrite(output_path, img0)

    return output_path, garbage_count, building_count
