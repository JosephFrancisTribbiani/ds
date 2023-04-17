import cv2
import cvzone
import logging
from ultralytics import YOLO


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

# approach 1 using webcam
# create webcam object
# cap = cv2.VideoCapture(0)  # webcam ID specified (it can be other digits if you are using multiple webcams)
# cap.set(3, 1280) # setting width (id 3)
# cap.set(4, 720)  # setting height (id 4)

# approach 2 for the video files
cap = cv2.VideoCapture("./Videos/cars.mp4")

# model initialization
model = YOLO(model="./Yolo_weights/yolov8n.pt")

while True:

    # launch webcam
    success, img = cap.read()

    # make prediction
    # stream=True means that we are using generator and it will be more efficient (recomended)
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:

            # approach 1 to draw bbox using xyxy format
            # x1, y1, x2, y2 = box.xyxy[0]  # xyxy returns the coordinates of two bounding box corners
            # x1, y1, x2, y2 = int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item())
            # LOGGER.debug("Bounding boxes coordinates")
            # LOGGER.debug("%s, %s, %s, %s" % (x1, y1, x2, y2))
            # # create rectangle of the bounding box on image
            # cv2.rectangle(img=img, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 255), thickness=3)  # img object, p1 coordinates, p2 coordinates, color, thickness

            # approach 2 to draw bounding box using xywh format
            x_center, y_center, w, h = box.xywh[0]
            x_center, y_center, w, h = int(x_center.item()), int(y_center.item()), int(w.item()), int(h.item())
            LOGGER.debug("Bounding boxes values (bbox center coordinates, width and height)")
            LOGGER.debug("%s, %s, %s, %s" % (x_center, y_center, w, h))
            # convert center coordinates to left corner coordinates
            x_left_corner, y_left_corner = x_center - w//2, y_center - h//2
            bbox = x_left_corner, y_left_corner, w, h
            # create rectangle of the bounding box on image
            cvzone.cornerRect(img=img, bbox=bbox)

            # get confidence
            conf = box.conf[0].item()
            LOGGER.debug("Confidence value: %s" % conf)
            
            # get class name
            class_idx = int(box.cls[0].item())
            class_name = CLASSNAMES[class_idx]
            LOGGER.debug("CLass ID: %s, class name: %s" % (class_idx, class_name))

            # plot confidence and class name
            cvzone.putTextRect(img=img, text="{} {:.2f}".format(class_name, conf), pos=(max(0, x_left_corner), max(35, y_left_corner - 10)), 
                               scale=1, thickness=1, offset=3, colorR=(124, 255, 0), colorT=(0, 0, 0))

    cv2.imshow("Image", img)
    cv2.waitKey(1)
