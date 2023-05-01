import os
import re
import cv2
import pickle
import logging
import numpy as np
import face_recognition
from typing import List, Union, Tuple
from pathlib import Path
from db import init_db


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")

ROOT = Path(__file__).resolve().parent
FACES_PATH = str(ROOT / "data" / "faces")
LOGGER = logging.getLogger(__name__)


def encode(face: np.ndarray) -> np.ndarray:
    # firstly we have to convert BGR (uses by cv2) to RGB (uses by face_recognition)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    # encode the image
    encoded = face_recognition.face_encodings(face)[0]
    return encoded


def main() -> None:

    # create tables if not exists
    init_db(force=True, hidden_size=128)
    
    # load faces and get IDs
    encodes = []
    for face in os.listdir(FACES_PATH):

        # getting face id
        id = int(re.search(r"[0-9]+", face).group(0))

        # loading face
        face_path = os.path.join(FACES_PATH, face)
        face = cv2.imread(face_path)

        # get encoding
        LOGGER.info("Encoding face with ID: %s" % id)
        enc = encode(face=face)
        encodes.append((id, enc))
        LOGGER.debug("Encoded face has shape: %s" % enc.shape)
    
    return

if __name__ == "__main__":
    main()
