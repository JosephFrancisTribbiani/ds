import os
import cv2
import sort
import cvzone
import logging
import numpy as np
from ultralytics import YOLO
from pathlib import Path


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s:%(message)s")
LOGGER = logging.getLogger(__name__)

# class names COCO
CLASSNAMES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
    ]
CURR_DIR = Path(__file__).resolve().parent
VIDEO_PATH = str(CURR_DIR.parent / "Videos" / "cars.mp4")
MODEL_PT = str(CURR_DIR.parent / "Yolo_weights" / "yolov8l.pt")
MASK_PATH  = str(CURR_DIR / "mask.jpg")
LIMITS = [220, 360, 780, 360]  # line between two point with format xyxy
MARGIN = 15

# counter of car
total_count = list()

# read the video file and mask
cap = cv2.VideoCapture(str(VIDEO_PATH))
mask = cv2.imread(MASK_PATH)

# tracking
# max_age - if for example ID 1 is lost on a frame, for how long (how many frames) should we wait it
tracker = sort.Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# model initialization
model = YOLO(model=MODEL_PT)

while True:

    # launch video and apply maks bitwise
    success, img = cap.read()
    img_region = cv2.bitwise_and(img, mask)

    # make prediction
    # stream=True means that we are using generator and it will be more efficient (recomended)
    results = model(img_region, stream=True)

    # create list of the detections
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes

        for box in boxes:
            # get class name
            class_idx = int(box.cls[0].item())
            class_name = CLASSNAMES[class_idx]
            LOGGER.debug("CLass ID: %s, class name: %s" % (class_idx, class_name))

            # get confidence
            conf = box.conf[0].item()
            LOGGER.debug("Confidence value: %s" % conf)

            # detect if truck, bus, car or motorbike and the confidence is more than 0.3
            if conf > 0.3 and class_name in ["truck", "bus", "car", "motorbike"]:
                # draw bounding box using xywh format
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item())
                LOGGER.debug("Bounding boxes values (xyxy format)")
                LOGGER.debug("%s, %s, %s, %s" % (x1, y1, x2, y2))
                
                # add detections
                curr_detection = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack([detections, curr_detection])

    LOGGER.debug("Tracking results:")
    tracker_results = tracker.update(detections)
    cv2.line(img=img, pt1=(LIMITS[0], LIMITS[1]), pt2=(LIMITS[2], LIMITS[3]), color=(0, 0, 255), thickness=5)

    for x1, y1, x2, y2, id in tracker_results:
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        LOGGER.debug("[x1, y1, x2, y2, id] - %s, %s, %s, %s, %s" % (x1, y1, x2, y2, id))

        # create rectangle of the bounding box on image
        bbox = x1, y1, x2 - x1, y2 - y1
        cvzone.cornerRect(img=img, bbox=bbox)

        # lets plot ID numbers
        cvzone.putTextRect(img=img, text="ID: %s" % id, pos=(max(0, x1), max(35, y1 - 10)), 
                           scale=1, thickness=1, offset=3, colorR=(124, 255, 0), colorT=(0, 0, 0))
        
        # center of the bbox and plot circles
        xc, yc = round((x1 + x2) / 2), round((y1 + y2) / 2)
        cv2.circle(img=img, center=(xc, yc), radius=5, color=(255, 0, 255), thickness=cv2.FILLED)

        if LIMITS[0] < xc < LIMITS[2] and LIMITS[1] - MARGIN < yc < LIMITS[3] + MARGIN:
            if total_count.count(id) == 0:
                total_count.append(id)

    # show total count
    cvzone.putTextRect(img=img, text="Cars: %s" % len(total_count), pos=(50, 50))

    cv2.imshow("Image", img)
    cv2.waitKey(1)
