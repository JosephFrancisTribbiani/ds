import os
import cv2
import sort
import cvzone
import logging
import numpy as np
from ultralytics import YOLO
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")
LOGGER = logging.getLogger(__name__)

PERSON_CLASS_NUM = 0
CURR_DIR = Path(__file__).resolve().parent
VIDEO_PATH = str(CURR_DIR.parent / "Videos" / "people.mp4")
MODEL_PT = str(CURR_DIR.parent / "Yolo_weights" / "yolov8n.pt")
MASK_PATH  = str(CURR_DIR / "mask.jpg")
OUTPUT = str(CURR_DIR / "output.mp4")

# define limits (lines between two point with format xyxy)
LIMITS_UP =   [103, 161, 296, 161]  # for people going up
LIMITS_DOWN = [527, 489, 735, 489]  # for people going down
MARGIN = 15

# counters of people
count_up = list()
count_down = list()

# read the video file and mask
cap = cv2.VideoCapture(str(VIDEO_PATH))
mask = cv2.imread(MASK_PATH)

# tracking
# max_age - if for example ID 1 is lost on a frame, for how long (how many frames) should we wait it
tracker = sort.Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# model initialization
model = YOLO(model=MODEL_PT)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(OUTPUT, fourcc, 30.0, (1280, 720))
while (cap.isOpened()):

    # launch video and apply maks bitwise
    success, img = cap.read()

    if not success:
        continue

    img_region = cv2.bitwise_and(img, mask)

    # show graphics
    img_graphics = cv2.imread(str(CURR_DIR / "graphics.png"), cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(imgBack=img, imgFront=img_graphics, pos=(730, 260))

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
            LOGGER.debug("CLass ID: %s" % class_idx)

            # get confidence
            conf = box.conf[0].item()
            LOGGER.debug("Confidence value: %s" % conf)

            # detect if person`s confidence is more than 0.3
            if conf > 0.3 and class_idx == 0:
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
    # cv2.line(img=img, pt1=(LIMITS_UP[0], LIMITS_UP[1]), pt2=(LIMITS_UP[2], LIMITS_UP[3]), color=(0, 0, 255), thickness=5)
    # cv2.line(img=img, pt1=(LIMITS_DOWN[0], LIMITS_DOWN[1]), pt2=(LIMITS_DOWN[2], LIMITS_DOWN[3]), color=(0, 0, 255), thickness=5)

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
        # cv2.circle(img=img, center=(xc, yc), radius=5, color=(255, 0, 255), thickness=cv2.FILLED)

        # count people going up
        if LIMITS_UP[0] < xc < LIMITS_UP[2] and LIMITS_UP[1] - MARGIN < yc < LIMITS_UP[3] + MARGIN:
            if count_up.count(id) == 0:
                count_up.append(id)

        # count people going down
        if LIMITS_DOWN[0] < xc < LIMITS_DOWN[2] and LIMITS_DOWN[1] - MARGIN < yc < LIMITS_DOWN[3] + MARGIN:
            if count_down.count(id) == 0:
                count_down.append(id)

    # show total count
    cv2.putText(img=img, text="%s" % len(count_up),   org=(929, 345),  fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=5, color=(139, 195, 75), thickness=7)
    cv2.putText(img=img, text="%s" % len(count_down), org=(1191, 345), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=5, color=(50, 50, 230),  thickness=7)

    out.write(img)
    cv2.imshow("Image", img)

    # выход при нажатии клавиши ESC
    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
