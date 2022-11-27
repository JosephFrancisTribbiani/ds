import re
import yaml
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from collections import defaultdict
from typing import Union, Dict, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.transforms import functional as F


class RunningLoss:
  def __init__(self, ):
    self.running_loss = defaultdict(int)
  
  def update(self, iter: int, **kwargs) -> None:
    for lossf, loss_value in kwargs.items():
      self.running_loss[lossf] = (self.running_loss[lossf] * iter + loss_value.cpu().item()) / (iter + 1)
    return

  def get_sum(self, ) -> float:
    loss_total = 0
    for _, loss in self.running_loss.items():
      loss_total += loss
    return loss_total


def load_dicom(directory: str) -> np.array:
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(directory)
    reader.SetFileNames(dicom_names)
    image_itk = reader.Execute()

    image_zyx = sitk.GetArrayFromImage(image_itk).astype(np.int16)
    return image_zyx


def load_mask(path: str):
  return np.rot90(nib.load(path).get_fdata().transpose(2, 0, 1), axes=(1, 2))


def min_max_norm(image: torch.tensor) -> torch.tensor:
  image -= image.min().item()
  image /= image.max().item()
  return image


class Rotation3DTransform(nn.Module):

  # dims:
  # 0: Z axis (plane (1, 2))
  # 1: Y axis (plane (0, 2))
  # 2: X axis (plane (0, 1))

  @staticmethod
  def get_params(angles: Dict = {0: (-15, 15), 1: (-8, 8), 2: (-8, 8)}):
    if angles is None:
      return None, None

    # select an axis
    dim = np.random.choice(list(angles.keys()))

    # sample the rotation angle
    angle = angles.get(dim)
    if isinstance(angle, list):
      angle = np.random.choice(angle)
    elif isinstance(angle, tuple) and len(angle) == 2:
      angle = np.random.uniform(*angle)
    return dim, angle
  
  def __init__(self):
    super().__init__()

  def __call__(self, image: torch.tensor, dim: int, angle: Union[float, int], slice_idx: int = None,
               interpolation: T.InterpolationMode = T.InterpolationMode.BILINEAR):
    if not angle:
      return image

    if dim:
      swap = (image.ndim - 3, image.ndim - 3 + dim)
      image = image.transpose(*swap)

    # define rotation center    
    height, width = image.shape[-2:]
    if dim == 1 and slice_idx is not None:
      center = [width // 2, slice_idx]
    elif dim == 2 and slice_idx is not None:
      center = [slice_idx, height // 2]
    else:
      center = [width // 2, height // 2]

    image = F.rotate(img=image, angle=angle, expand=False, center=center,
                     interpolation=interpolation, fill=image.min().item())
    
    if dim:
      image = image.transpose(*swap)
    return image


def read_yaml(file_path: str) -> dict:
  """
  Функция для считывания config файла.
  :param file_path: путь к config файлу.
  :return: параметры из config файла.
  """

  with open(file_path, 'r') as f:
    try:
      loader = yaml.SafeLoader

      # добавляем возможность считывать числа, записанные в YAML файл в формате 1e-4, к примеру
      loader.add_implicit_resolver(
          u'tag:yaml.org,2002:float',
          re.compile(u'''^(?:
                         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                         |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                         |\\.[0-9_]+(?:[eE][-+][0-9]+)?
                         |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
                         |[-+]?\\.(?:inf|Inf|INF)
                         |\\.(?:nan|NaN|NAN))$''', re.X),
          list(u'-+0123456789.')
      )

      return yaml.load(f, Loader=loader)
    except yaml.YAMLError as exc:
      print(exc)


@dataclass
class ModelConfig:
  encoder_name: str = "resnet34"
  encoder_weights: str = "imagenet"
  in_channels: int = 3
  classes: int = 1


@dataclass
class DataLoaderConfig:
  batch_size: int = 32
  num_workers: int = 0
  pin_memory: bool = False


@dataclass
class DataSetConfig:
  metadata_loc: str = "./data/metadata.json"
  transform: bool = True
  crop_size: int = 224
  in_channels: int = 3
  loc: List[float] = None
  scale: List[float] = None
  p_3drot: float = 0.5
  p_2drot: float = 0.5
  p_rcrop: float = 0.9
  p_hflip: float = 0.5
