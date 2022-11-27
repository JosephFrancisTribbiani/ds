# Тестовое задание

[Ссылка](https://lightningseas.notion.site/Junior-CV-Engineer-5995a1401f3d4316a13051f7aa6a6d53#987dc626d5914b298e1e20fde140ce99) на тестовое задание. Код нужно закинуть в репо на гитхаб и приложить ссылку в эту [форму](https://airtable.com/shrXqH5jJaSM2a3Uv)

## Описание задания

Нужно написать бейзлайн для обучения сегментационной нейросети для обнаружения плеврального выпота. 

Обучить можно любую сегментационную нейросеть, можно использовать 2D или 3D архитектуру.

Код должен быть разделен следующим образом:

- файл с архитектурой модели
- файл с препроцессингом данных для обучения
- файл с датасетом
- файл с функцией подсчета [DICE Coef](https://radiopaedia.org/articles/dice-similarity-coefficient#:~:text=The%20Dice%20similarity%20coefficient%2C%20also,between%20two%20sets%20of%20data.)
- основной файл с самим циклом обучения. В коде обучения во время валидации автоматически должна  выбираться лучшая эпоха и сохраняться веса модели в папку `output`. Также после цикла обучения в эту папку должна сохраняться картинка с изменениями коэффициента DICE с каждой эпохой. (по оси Y - коэффициент DICE, по оси X - номер эпохи). Вместо выходной картинки с графиком можно использовать любые трекеры при желании (tensorboard etc.)

## Описание данных

Данные лежат в этом [архиве](./data/subset.zip)

Внутри два архива - `subset_img.zip` и `subset_masks.zip`. 

В `subset_img.zip` лежат следующие папки

```
LUNG1-001	LUNG1-002	LUNG1-005	LUNG1-008	LUNG1-013	LUNG1-016	LUNG1-018	LUNG1-024	LUNG1-026	LUNG1-028
```

Для чтения данных оттуда можно использовать следующую функцию 

```python
import SimpleITK as sitk

def load_dicom(directory):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(directory)
    reader.SetFileNames(dicom_names)
    image_itk = reader.Execute()

    image_zyx = sitk.GetArrayFromImage(image_itk).astype(np.int16)
    return image_zyx 
```

Функция возвращает 3д куб в формате `numpy array`

`subset_masks.zip` лежат следующие папки

```
LUNG1-001	LUNG1-002	LUNG1-005	LUNG1-008	LUNG1-013	LUNG1-016	LUNG1-018	LUNG1-024	LUNG1-026	LUNG1-028
```

Внутри каждой из них лежит файл формата `.nii.gz` - в них лежат бинарные сегментационные маски

По именам папки из `masks` соответствуют именам папок из `img`

Для чтения файлов можно использовать следующий сниппет кода

```python
import nibabel as nib

mask = nib.load(tmp_m)
mask = mask.get_fdata().transpose(2, 0, 1)
```
После этого в переменной `mask` будет лежать `numpy array`, который по размерностям совпадает с `image_zyx`