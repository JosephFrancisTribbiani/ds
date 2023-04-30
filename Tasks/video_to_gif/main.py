import cv2
import glob
import os
import shutil
import PySimpleGUI as sg

from PIL import Image

file_types = [("MP4 (*.mp4)", "*.mp4"), ("All files (*.*)", "*.*")]


def convert_mp4_to_jpgs(path: str, step: int = 1, scale: int = None) -> None:
    if os.path.exists("./output"):
        # remove previous GIF frame files
        shutil.rmtree("./output")

    try:
        os.mkdir("./output")

    except IOError:
        sg.popup("Error occurred creating output folder")
        return
    
    n = 0
    cap = cv2.VideoCapture(path)
    while True:
        success, frame = cap.read()

        if not success:
            break

        if not n % step:
            if scale is not None:
                w = int(frame.shape[1] * scale / 100)
                h = int(frame.shape[0] * scale / 100)
                frame = cv2.resize(frame, (w, h), interpolation = cv2.INTER_AREA)

            cv2.imwrite(f"./output/frame_{n:05d}.jpg", frame)

        n += 1
    return


def make_gif(gif_path: str, frame_folder: str = "output") -> None:
    images = glob.glob(f"{frame_folder}/*.jpg")
    images.sort()
    frames = [Image.open(image) for image in images]
    frame_one = frames[0]
    frame_one.save(gif_path, format="GIF", append_images=frames,
                   save_all=True, duration=50, loop=0)
    return


def main() -> None:

    # диалоговое окно
    layout = [
        [
            sg.Text("MP4 File Location", size=(20, 1)),
            sg.Input(size=(30, 1), key="-FILENAME-", disabled=True),
            sg.FileBrowse(file_types=file_types),
        ],
        [
            sg.Text("GIF File Save Location", size=(20, 1)),
            sg.Input(size=(30, 1), key="-OUTPUTFILE-", disabled=True),
            sg.SaveAs(file_types=file_types),
        ],
        [
            sg.Text("Frame Step (Percent)", size=(20, 1)),
            sg.Input(size=(5, 1), key="-STEP-", disabled=False),
            sg.Text("Resize Scale", size=(10, 1)),
            sg.Input(size=(5, 1), key="-SCALE-", disabled=False)
        ],
        [sg.Button("Convert to GIF")],
    ]

    window = sg.Window("MP4 to GIF Converter", layout)

    while True:
        event, values = window.read()
        mp4_path = values["-FILENAME-"]
        gif_path = values["-OUTPUTFILE-"]
        step     = int(values["-STEP-"])
        
        if values["-SCALE-"].isdigit():
            scale = int(values["-SCALE-"])

        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        if event in ["Convert to GIF"]:
            if mp4_path and gif_path:
                convert_mp4_to_jpgs(path=mp4_path, step=step, scale=scale)
                make_gif(gif_path=gif_path)
                shutil.rmtree("./output")
                sg.popup(f"GIF created: {gif_path}")

    window.close()
    return


if __name__ == "__main__":
    main()
