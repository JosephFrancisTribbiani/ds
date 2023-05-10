import os
import cv2
import cvzone
import logging
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict
from poker_hand_function import CheckHand, parse_card


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")
LOGGER = logging.getLogger(__name__)

CURR_DIR = Path(__file__).resolve().parent
MODEL_PT = str(CURR_DIR / "results" / "detect" / "train" / "weights" / "best.pt")
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
CARDS = {
    "ranks": {
        "A": "ace",
        "J": "jack",
        "Q": "queen",
        "K": "king",
    },
    "suits": {
        "C": "clubs",
        "D": "diamonds",
        "H": "hearts",
        "S": "spades",
    },
    "loc": str(CURR_DIR / "cards")
}


def get_card_path(card: str) -> str:
    rank, suit = parse_card(card=card)
    rank, suit = CARDS["ranks"][rank] if isinstance(rank, str) else rank, CARDS["suits"][suit]
    fpath = os.path.join(CARDS["loc"], "%s_of_%s.png" % (rank, suit))
    return fpath


def main():
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
        ch = CheckHand()

        # overlay graphics
        img_graphics = cv2.imread(str(CURR_DIR / "graphics.png"), cv2.IMREAD_UNCHANGED)
        img = cvzone.overlayPNG(imgBack=img, imgFront=img_graphics, pos=(0, 0))

        for r in results:
            boxes = r.boxes
            hand = defaultdict(list)

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

                hand[class_name].append(x_center)

                # plot confidence and class name
                cvzone.putTextRect(img=img, text="{} {:.2f}".format(class_name, conf), pos=(max(0, x_left_corner), max(35, y_left_corner - 10)), 
                                   scale=1, thickness=1, offset=3, colorR=(124, 255, 0), colorT=(0, 0, 0))
            
            hand = [(card, sum(pos) / len(pos)) for card, pos in hand.items()]
            hand.sort(key=lambda val: val[1])
            hand = [card for card, _ in hand]
            for idx in range(min(5, len(hand))):
                card = hand[idx]
                x_pos = 310 + idx*140
                card_loc = get_card_path(card=card)
                img_card = cv2.imread(card_loc, cv2.IMREAD_UNCHANGED)
                img_card = cv2.resize(img_card, dsize=(100, 145), interpolation=cv2.INTER_LINEAR)
                img = cvzone.overlayPNG(imgBack=img, imgFront=img_card, pos=(x_pos, 543))

            if len(hand) == 5:
                comb = ch(hand=hand)
                comb = comb.upper()
                cv2.putText(img=img, text=comb, org=(15, 36), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1.2, color=(255, 0, 0), thickness=1, )
                LOGGER.info("Текущая комбинация: %s" % comb)

        cv2.imshow("Image", img)

        # выход при нажатии клавиши ESC
        if cv2.waitKey(1) == 27:  # 27 - escape key
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
