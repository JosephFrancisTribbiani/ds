import os
import cvzone
import logging
import cv2 as cv
import numpy as np
import face_recognition
from pathlib import Path
from db import read_embeddings


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")
LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent
BACKGROUND_PATH = str(ROOT / "data" / "background.png")
MODES_PATH = str(ROOT / "data" / "modes")
SCALE = 0.25
TOL = 0.6


def main() -> None:
    cap = cv.VideoCapture(0)  # create webcam object
    cap.set(3, 640)  # set width
    cap.set(4, 480)  # set hight

    # loading images (background and for modes)
    background = cv.imread(BACKGROUND_PATH)
    modes = [cv.imread(os.path.join(MODES_PATH, mode)) for mode in os.listdir(MODES_PATH)]

    # load all embeddings
    LOGGER.info("Loading embeddings")
    known_face_metadata, known_face_encodings = read_embeddings(hidden_size=128)
    LOGGER.info("Loaded %s embeddings" % len(known_face_encodings))

    while True:
        
        # read frame from webcap object
        success, frame = cap.read()

        if not success:
            continue

        # resize image from webcam to save computation resources
        # and convert to RGB from BGR
        frame_s = cv.resize(frame, (0, 0), None, SCALE, SCALE)
        frame_s = cv.cvtColor(frame_s, cv.COLOR_BGR2RGB)

        # находим все лица на изображении и кодируем
        face_curr_locations = face_recognition.face_locations(frame_s)
        face_encodings_to_check = face_recognition.face_encodings(frame_s, known_face_locations=face_curr_locations)
        LOGGER.info("ОБнаружено лиц: %s" % len(face_encodings_to_check))

        # сравниваем каждое обнаруженное лицо с выгруженными эмбеддингами
        for face_encoding_to_check, face_curr_location in zip(face_encodings_to_check, face_curr_locations):
            matches = face_recognition.compare_faces(
                known_face_encodings=known_face_encodings, face_encoding_to_check=face_encoding_to_check, tolerance=TOL)
            dist    = face_recognition.face_distance(face_encodings=known_face_encodings, face_to_compare=face_encoding_to_check)

            # ищем ближайшее лицо
            clst_idx = np.argmin(dist)
            if matches[clst_idx]:
                color = (0, 255, 0)
                clst_meta = known_face_metadata[clst_idx]
                firstname, secondname = clst_meta.get("firstname"), clst_meta.get("secondname")
                firstname, secondname = firstname.capitalize(), secondname.capitalize()
                text = "%s %s" % (firstname, secondname)
            else:
                color = (0, 0, 255)
                text   = "Unknown"

            # plot bounding box
            bottom, right, top, left = face_curr_location
            x1, y1 = round(left / SCALE), round(bottom / SCALE)
            w, h = round((right - left) / SCALE), round((top - bottom) / SCALE)
            bbox = (x1, y1, w, h)
            cvzone.cornerRect(img=frame, bbox=bbox, colorC=color)
            cvzone.putTextRect(img=frame, text=text, pos=(max(0, x1), max(35, y1 - 10)), 
                               scale=1, thickness=1, offset=3, colorR=color, colorT=(0, 0, 0))
        
        # overlay background, mode with webcam 
        background[162:162 + 480, 55:55 + 640] = frame
        background[44:44 + 633, 808:808 + 414] = modes[3]

        # exit due to ESC key
        if cv.waitKey(1) == 27:
            break
        
        cv.imshow("Face Attendance", background)

    cv.destroyAllWindows()

    return


if __name__ == "__main__":
    main()