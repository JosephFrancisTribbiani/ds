import cv2
from ultralytics import YOLO


model = YOLO(model="../Yolo_weights/yolov8l.pt")  # yolo8n means nano, yolo8l means large
results = model("./Images/3.png", show=True)
cv2.waitKey(0)
