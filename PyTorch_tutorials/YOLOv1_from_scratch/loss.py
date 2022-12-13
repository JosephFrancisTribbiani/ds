import torch
import torch.nn as nn
from utils import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self, s: int = 7, b: int = 2, c: int = 20) -> None:
        """
        :param s: split size (number of cells is s**2)
        :param b: numeber of anchors per cell
        :param c: number of classes
        """
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.s = s
        self.b = b
        self.c = c
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, prediction: torch.tensor, target: torch.tensor):
        prediction = prediction.reshape(-1, self.s, self.s, self.b*5 + self.c)

        # predicts looks for each cell:
        # [x, y, w, h, p, x, y, w, h, p, ..., probabilities for classes]
        # x and y coordinates from top left cell corner (relative, means that values from [0, 1])
        # h and w - height and width respectively
        # where p - confidence, that this bounding box contains object
        # 
        # calculates IoU for each cell
        # you penalize only for bounding box which has biggest IoU between predicted and ground truth
        # and object exists in cell
        ious = torch.cat(
            [intersection_over_union(prediction[..., i:i + 4], target[..., i:i + 4]).unsqueeze(0) \
                for i in range(self.c + 1, self.c + self.b*5, 5)], dim=0)
        ious_maxes, bestbox = torch.max(ious, dim=0)
        return
        