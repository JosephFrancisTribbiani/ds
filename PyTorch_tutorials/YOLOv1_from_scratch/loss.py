import torch
import torch.nn as nn
from typing import Union, Tuple
from utils import get_best_box, extend_target, get_mask


class YoloLoss(nn.Module):
    def __init__(self, s: int = 7, b: int = 2, lambda_noobj: Union[float, int] = 0.5, 
                 lambda_coord: Union[int, float] = 5) -> None:
        """
        :param s: split size (number of cells is s**2)
        :param b: numeber of anchors per cell
        """
        super().__init__()
        self.s = s
        self.b = b
        self.lambda_noobj = lambda_noobj
        self.lambda_coord = lambda_coord
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, prediction: torch.tensor, target: torch.tensor) -> torch.tensor:
        """
        YOLO v1 loss calculation.
        :param prediction: 2D tensor shape [batch_size*grid_size*grid_size, 5*n_anchors + n_classes]
        :param target: 2D tensor shape [batch_size*grid_size*grid_size, 5 + n_classes]
        """
        # First of all, we have to calculate IoU between all anchor boxes and all bounding boxes
        # for each cell and find index of anchor box with highest IoU value.
        # YOLO predict square root of width and height. Also this values are relative to the image size.
        # So before IoU calculation we have to modify them, as well as target values
        ious_maxes, bestbox = get_best_box(prediction=prediction, target=target, n_anchors=self.b, grid_size=self.s)

        # Then we have to prepare target vector (same length as prediction vector).
        target_ext = extend_target(target=target, bestbox=bestbox, bestbox_iou=ious_maxes, n_anchors=self.b)

        # mse loss calculation
        loss = self.mse(prediction, target_ext)

        # If object does not exists in the cell, we do not penalize coordinates and classes probabilities
        # but only confidence values. So we have to mask our loss function.
        mask = get_mask(size=loss.shape, has_object=target[..., [0]], bestbox=bestbox, n_anchors=self.b, 
                        lambda_coord=self.lambda_coord, lambda_noobj=self.lambda_noobj)

        # final loss calculation
        return torch.sum(loss*mask)
