import os
import cvzone
import logging
import cv2 as cv
import numpy as np
import face_recognition
from pathlib import Path
from db import read_embeddings, mark_students, get_metadata


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")
LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent
BACKGROUND_PATH = str(ROOT / "data" / "background.png")
MODES_PATH = str(ROOT / "data" / "modes")
SCALE = 0.25
TOL = 0.55


def compare_embs(face_encodings_to_check: list, hidden_size: int = 128) -> int:
    # load all embeddings
    LOGGER.info("Loading embeddings")
    known_faces          = read_embeddings(hidden_size=hidden_size)
    known_face_ids       = list(known_faces.keys())
    known_face_encodings = list(known_faces.values())
    LOGGER.info("Loaded %s embeddings" % len(known_faces))

    for face_encoding_to_check in face_encodings_to_check:
        matches = face_recognition.compare_faces(
            known_face_encodings=known_face_encodings, face_encoding_to_check=face_encoding_to_check, tolerance=TOL)
        dist = face_recognition.face_distance(face_encodings=known_face_encodings, face_to_compare=face_encoding_to_check)

        # ищем ближайшее лицо
        clst_idx = np.argmin(dist)
        if matches[clst_idx]:
            id = known_face_ids[clst_idx]
        else:
            id = -1
        yield id


def main() -> None:
    cap = cv.VideoCapture(0)  # create webcam object
    cap.set(3, 640)  # set width
    cap.set(4, 480)  # set hight

    # loading images (background and for modes)
    background = cv.imread(BACKGROUND_PATH)
    modes = [cv.imread(os.path.join(MODES_PATH, mode)) for mode in os.listdir(MODES_PATH)]

    while True:
        
        # read frame from webcap object
        success, frame = cap.read()

        if not success:
            continue

        # resize image from webcam to save computation resources
        # and convert to RGB from BGR
        frame_s = cv.resize(frame, (0, 0), None, SCALE, SCALE)
        frame_s = cv.cvtColor(frame_s, cv.COLOR_BGR2RGB)

        # находим все лица на изображении и получаем для них encodings
        face_curr_locations     = face_recognition.face_locations(frame_s)
        face_encodings_to_check = face_recognition.face_encodings(frame_s, known_face_locations=face_curr_locations)
        LOGGER.info("Обнаружено лиц: %s" % len(face_encodings_to_check))

        # сравниваем каждое обнаруженное лицо с выгруженными эмбеддингами
        # и получаем список обнаруженных ID
        # если ID == -1, то данное лицо сервис не знает
        recognized_ids = [id for id in compare_embs(face_encodings_to_check=face_encodings_to_check)]

        # далее нам необходимо отметить студентов в таблице attendance
        # но вначале необходимо отфильтровать -1 от остальных ID в recognized_ids
        # mark_students возвращает список ID, которые были отмечены 
        # (т.е. с момента их последнего посещения прошло более 30-ти минут)
        known_recognized_ids = list(filter(lambda id: id >= 0, recognized_ids))
        known_faces_marked   = mark_students(ids=known_recognized_ids)

        # по обнаруженным лицам, о которых знает сервис, выгрузим метаданные
        # и отметим последнее посещение
        known_recognized_meta = get_metadata(ids=known_recognized_ids)
        for id in known_faces_marked:
            known_recognized_meta[id]["marked"] = True

        # далее итериуемся по bounding boxes обнаруженных лиц
        # и отмечаем на панели, предварительно отсортировав их по ID
        bboxes_to_plot = list(zip(recognized_ids, face_curr_locations))
        bboxes_to_plot.sort(key=lambda val: val[0])
        for id, bbox in bboxes_to_plot:
            bottom, right, top, left = bbox
            x1, y1 = round(left / SCALE), round(bottom / SCALE)
            w, h   = round((right - left) / SCALE), round((top - bottom) / SCALE)
            bbox   = (x1, y1, w, h)

            if isinstance(known_recognized_meta, dict) and known_recognized_meta.get(id) is not None:
                firstname = known_recognized_meta[id].get("firstname")
                secondname = known_recognized_meta[id].get("secondname")
                firstname, secondname = firstname.capitalize(), secondname.capitalize()
                text = "%s %s" % (firstname, secondname)
                color = (0, 255, 0)
            else:
                text  = "Unknown"
                color = (0, 0, 255)

            cvzone.cornerRect(img=frame, bbox=bbox, colorC=color)
            cvzone.putTextRect(img=frame, text=text, pos=(max(0, x1), max(35, y1 - 10)), 
                               scale=1, thickness=1, offset=3, colorR=color, colorT=(0, 0, 0))

        # overlay background with webcam 
        background[162:162 + 480, 55:55 + 640] = frame

        # exit due to ESC key
        if cv.waitKey(1) == 27:
            break
        
        cv.imshow("Face Attendance", background)

    cv.destroyAllWindows()

    return


if __name__ == "__main__":
    main()