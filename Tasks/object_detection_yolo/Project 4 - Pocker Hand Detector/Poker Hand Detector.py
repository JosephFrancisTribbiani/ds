import cv2
import cvzone
import logging
from pathlib import Path
from ultralytics import YOLO


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s:%(message)s")
LOGGER = logging.getLogger(__name__)

# class names COCO
CLASSNAMES = [
    "10C", "10D", "10H", "10S", 
    "2C", "2D", "2H", "2S", 
    "3C", "3D", "3H", "3S", 
    "4C", "4D", "4H", "4S", 
    "5C", "5D", "5H", "5S", 
    "6C", "6D", "6H", "6S", 
    "7C", "7D", "7H", "7S", 
    "8C", "8D", "8H", "8S", 
    "9C", "9D", "9H", "9S", 
    "AC", "AD", "AH", "AS", 
    "JC", "JD", "JH", "JS", 
    "KC", "KD", "KH", "KS", 
    "QC", "QD", "QH", "QS"
    ]
CURR_DIR = Path(__file__).resolve().parent
MODEL_PT = str(CURR_DIR / "runs" / "detect" / "train" / "weights" / "best.pt")

# create webcam object
cap = cv2.VideoCapture(0)  # webcam ID specified (it can be other digits if you are using multiple webcams)
cap.set(3, 1280) # setting width (id 3)
cap.set(4, 720)  # setting height (id 4)

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

            # draw bounding box using xywh format
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
