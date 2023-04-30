import cv2
import cvzone
import logging
from pathlib import Path
from ultralytics import YOLO


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s:%(message)s")
LOGGER = logging.getLogger(__name__)

# class names COCO
CLASSNAMES = [
    "Excavator", "Gloves", "Hardhat", "Ladder", "Mask", "NO-Hardhat", "NO-Mask", 
    "NO-Safety Vest", "Person", "SUV", "Safety Cone", "Safety Vest", "bus", 
    "dump truck", "fire hydrant", "machinery", "mini-van", "sedan", "semi", "trailer", 
    "truck and trailer", "truck", "van", "vehicle", "wheel loader"
    ]
CURR_DIR = Path(__file__).resolve().parent
VIDEO_PATH = str(CURR_DIR.parent / "Videos" / "ppe-2.mp4")
MODEL_PT = str(CURR_DIR / "runs" / "detect" / "train" / "weights" / "best.pt")
WEBCAM = False

# approach 1 using webcam
if WEBCAM:
    # create webcam object
    cap = cv2.VideoCapture(0)  # webcam ID specified (it can be other digits if you are using multiple webcams)
    cap.set(3, 1280) # setting width (id 3)
    cap.set(4, 720)  # setting height (id 4)

else:
    # approach 2 for the video files
    cap = cv2.VideoCapture(VIDEO_PATH)

# model initialization
model = YOLO(MODEL_PT)

while True:

    # launch
    success, img = cap.read()

    if not success:
        continue

    # make prediction
    # stream=True means that we are using generator and it will be more efficient (recomended)
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes

        for box in boxes:

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

    # выход при нажатии клавиши ESC
    if cv2.waitKey(1) == 27:  # 27 - escape key
        break

cv2.destroyAllWindows()
