import os
import json
import logging
import PySimpleGUI as sg
from db import get_next_id, save_student
from PIL import Image
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")
LOGGER = logging.getLogger(__name__)
FILE_TYPES = [
    ("PNG (*.png)", ["*.png", "*.PNG"]), 
    ("JPG (*.jpg)", ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]), 
    ("All files (*.*)", "*.*")]
ROOT = Path(__file__).resolve().parent
LAYOUT = [
    [
        sg.Text("Имя:", size=(15, 1)),
        sg.Input(size=(30, 1), key="-FIRSTNAME-")
    ],
    [
        sg.Text("Фамилия:", size=(15, 1)),
        sg.Input(size=(30, 1), key="-SECONDNAME-")
    ],
    [
        sg.Text("Статус:", size=(15, 1)),
        sg.Input(default_text="G", size=(30, 1), key="-STANDING-")
    ],
    [
        sg.Text("Специализация:", size=(15, 1)),
        sg.Input(size=(30, 1), key="-MAJOR-")
    ],
    [
        sg.Text("Начало обучения:", size=(15, 1)),
        sg.Input(size=(30, 1), key="-YEAR-")
    ],
    [
        sg.Text("Фотография", size=(15, 1)),
        sg.Input(size=(30, 1), key="-PHOTO-", disabled=True),
        sg.FileBrowse(file_types=FILE_TYPES, size=(8, 1), button_text="Открыть")
    ],
    [
        sg.Text("Записать в", size=(15, 1)),
        sg.Input(size=(30, 1), key="-SAVELOC-", disabled=True, default_text=str(ROOT / "data")),
        sg.FolderBrowse(size=(8, 1), button_text="Открыть")
    ],
    [
        sg.Button("Сохранить", size=(12, 1)), sg.Button("Закрыть", size=(12, 1))
    ],
    [
        sg.StatusBar("", size=(0, 1), key='-STATUS-')
    ],
]


def main() -> None:
    window = sg.Window("Форма для заполнения", LAYOUT)
    while True:
        event, values = window.read()

        if event == "Закрыть" or event == sg.WIN_CLOSED:
            break

        else:
            # считываем поля
            firstname = values["-FIRSTNAME-"]
            secondname = values["-SECONDNAME-"]
            standing = values["-STANDING-"]
            major = values["-MAJOR-"]
            starting_year = values["-YEAR-"]
            photo_loc = values["-PHOTO-"]
            save_loc = values["-SAVELOC-"]

            if event == "Сохранить":

                # проверяем, что все поля заполнены
                if (isinstance(firstname, str) and len(firstname) > 0) and \
                   (isinstance(secondname, str) and len(secondname) > 0) and \
                   (isinstance(standing, str) and len(standing) == 1) and \
                   (isinstance(major, str) and len(major) > 0) and \
                   (isinstance(starting_year, str) and starting_year.strip().isdecimal()) and \
                   (isinstance(photo_loc, str) and os.path.exists(photo_loc)):

                    # значения делаем upper case
                    firstname = firstname.strip().upper()
                    secondname = secondname.strip().upper()
                    standing = standing.strip().upper()
                    major = major.strip().upper()
                    starting_year = int(starting_year.strip())

                    # выводим результат
                    LOGGER.debug("Имя: %s" % firstname)
                    LOGGER.debug("Фамилия: %s" % secondname)
                    LOGGER.debug("Статус: %s" % standing)
                    LOGGER.debug("Специализация: %s" % major)
                    LOGGER.debug("Начало обучения: %s" % starting_year)

                    # если нет пути для сохранения, то создаем
                    if not os.path.exists(save_loc):
                        LOGGER.debug("Создаем папку %s" % save_loc)
                        os.makedirs(save_loc)

                    photo_path = os.path.join(save_loc, "photo")
                    if not os.path.exists(photo_path):
                        LOGGER.debug("Создаем папку %s" % photo_path)
                        os.makedirs(photo_path)

                    metadata_path = os.path.join(save_loc, "metadata")
                    if not os.path.exists(metadata_path):
                        LOGGER.debug("Создаем папку %s" % metadata_path)
                        os.makedirs(metadata_path)

                    # получаем следующий свободный ID студента
                    LOGGER.debug("Получаем ID студента")
                    id = get_next_id()
                    LOGGER.debug("ID: %s" % id)

                    # сохраняем метадату по студенту
                    LOGGER.debug("Сохраняем метаданные")
                    metadata = dict(
                        id=id,
                        firstname=firstname,
                        secondname=secondname,
                        standing=standing,
                        major=major,
                        starting_year=starting_year,
                    )
                    with open(os.path.join(metadata_path, "%s.json") % id, "w") as json_f:
                        json.dump(metadata, json_f)

                    # сохраняем фотографию
                    LOGGER.debug("Сохраняем фотографию")
                    _, ext = os.path.splitext(photo_loc)
                    photo = Image.open(photo_loc)
                    photo = photo.resize((216, 216), resample=Image.BILINEAR)
                    photo.save(os.path.join(photo_path, "%s%s" % (id, ext)))

                    # записываем в базу
                    save_student(**metadata)
                    LOGGER.debug("Данные сохранены")

                    state = "Данные сохранены"
                
                else:
                    state = "Поля не заполнены / запонены не верно"

                window['-STATUS-'].update(state)             
    
    window.close()
    return


if __name__ == "__main__":
    main()
