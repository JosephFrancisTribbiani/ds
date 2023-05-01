import os
import cv2
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BACKGROUND_PATH = str(ROOT / "data" / "background.png")
MODES_PATH = str(ROOT / "data" / "modes")

def main() -> None:
    cap = cv2.VideoCapture(0)  # create webcam object
    cap.set(3, 640)  # set width
    cap.set(4, 480)  # set hight

    # loading images (background and for modes)
    background = cv2.imread(BACKGROUND_PATH)
    modes = [cv2.imread(os.path.join(MODES_PATH, mode)) for mode in os.listdir(MODES_PATH)]

    while True:
        
        # read frame from webcap object
        success, frame = cap.read()

        if not success:
            continue
        
        # overlay background, mode with webcam 
        background[162:162 + 480, 55:55 + 640] = frame
        background[44:44 + 633, 808:808 + 414] = modes[3]

        # exit due to ESC key
        if cv2.waitKey(1) == 27:
            break
        
        cv2.imshow("Face Attendance", background)

    cv2.destroyAllWindows()

    return


if __name__ == "__main__":
    main()