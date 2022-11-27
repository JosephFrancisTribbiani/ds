import os
import json
import random
import pandas as pd
from typing import Tuple
from .utils import Rotation3DTransform, min_max_norm, load_dicom, load_mask

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


class AiDiagnosticDataset(Dataset):
  def __init__(self, metadata_loc: str = "./data/metadata.json", transform: bool = True, crop_size: int = 224, 
               in_channels: int = 3, loc: Tuple[float] = None, scale: Tuple[float] = None, p_3drot: float = 0.5,
               p_2drot: float = 0.5, p_rcrop: float = 0.9, p_hflip: float = 0.5) -> None:
    self.transform_ = transform
    self.crop_size = crop_size
    self.in_channels = in_channels
    self.loc = loc
    self.scale = scale
    
    # transformation probabilities
    self.p_3drot = p_3drot
    self.p_2drot = p_2drot
    self.p_rcrop = p_rcrop
    self.p_hflip = p_hflip

    # read metadata from file
    with open(metadata_loc, "r") as json_file:
        self.metadata = json.load(json_file)
        self.metadata = {int(k): v for k, v in self.metadata.items()}

    # get list of all slices
    self.slices = pd.DataFrame(data=[(k, range(len(os.listdir(v.get("img").replace("\\", "/"))))) for k, v in self.metadata.items()],
                               columns=["lung", "slice"]) \
                    .explode("slice") \
                    .reset_index(drop=True, inplace=False)

  @staticmethod
  def transform(image: torch.tensor, mask: torch.tensor, slice_idx: int = None, crop_size: int = 224, 
                in_channels: int = 3, loc: Tuple[float] = None, scale: Tuple[float] = None,
                p_3drot: float = 0.5, p_2drot: float = 0.5, p_rcrop: float = 0.9, p_hflip: float = 0.5) -> Tuple[torch.tensor, torch.tensor]:
    # augmentarion methods
    # 3D rotation
    if random.random() < p_3drot:
      rotation_transform = Rotation3DTransform()
      dim, angle = rotation_transform.get_params(angles={1: (-8, 8), 2: (-8, 8)})
      image = rotation_transform(image=image, dim=dim, angle=angle, slice_idx=slice_idx)
      mask = rotation_transform(image=mask, dim=dim, angle=angle, slice_idx=slice_idx,
                                interpolation=T.InterpolationMode.NEAREST)
    image, mask = \
      image[slice_idx, :, :].detach().unsqueeze(0), mask[slice_idx, :, :].detach().unsqueeze(0)
    
    
    # 2D rotation of slice
    if random.random() < p_2drot:
      degrees = T.RandomRotation.get_params(degrees=(-15, 15))
      image = F.rotate(img=image, angle=degrees, expand=False, interpolation=T.InterpolationMode.BILINEAR, 
                       fill=image.min().item())
      mask = F.rotate(img=mask, angle=degrees, expand=False, interpolation=T.InterpolationMode.NEAREST, 
                      fill=mask.min().item())

    # random crop
    if random.random() < p_rcrop:
      i, j, h, w = T.RandomCrop.get_params(
          image, output_size=(408, 408))
      image = F.crop(image, i, j, h, w)
      mask = F.crop(mask, i, j, h, w)

    # random horizontal flip
    if random.random() < p_hflip:
      image = F.hflip(image)
      mask = F.hflip(mask)

    # Resize
    image, mask = \
      F.resize(image, size=crop_size), F.resize(mask, size=crop_size, 
                                                interpolation=T.InterpolationMode.NEAREST)

    # rescale to [0, 1] and extend N chanels if needed
    image = min_max_norm(image)
    if in_channels > 1:
      image = image.repeat(in_channels, 1, 1)

    # image normalization
    if (loc is not None and scale is not None) and (len(loc) == in_channels and len(scale) == in_channels):
      image = F.normalize(image, mean=loc, std=scale)
    else:
      image = F.normalize(image, mean=image.mean(dim=(1, 2)), std=image.std(dim=(1, 2)))
    return image, mask

  def __len__(self):
    return self.slices.shape[0]
 
  def __getitem__(self, index):
    lung_num, slice_idx = self.slices.loc[index, ["lung", "slice"]]

    # load data
    img_loc = self.metadata.get(lung_num).get("img").replace("\\", "/")
    mask_loc = self.metadata.get(lung_num).get("mask").replace("\\", "/")
    image, mask = load_dicom(directory=img_loc), load_mask(path=mask_loc)

    # convert to tensor
    image, mask = \
      torch.tensor(image.copy(), dtype=torch.float32), torch.tensor(mask.copy(), dtype=torch.float32)
    
    # transform loaded data
    if not self.transform_:
      return image[slice_idx, :, :].detach().unsqueeze(0), mask[slice_idx, :, :].detach().unsqueeze(0)

    image, mask = self.transform(image=image, mask=mask, slice_idx=slice_idx, crop_size=self.crop_size, 
                                 in_channels=self.in_channels, loc=self.loc, scale=self.scale, p_3drot=self.p_3drot, 
                                 p_2drot=self.p_2drot, p_rcrop=self.p_rcrop, p_hflip=self.p_hflip)
    return image, mask
    