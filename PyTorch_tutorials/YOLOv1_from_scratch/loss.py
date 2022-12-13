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
        """
        :param prediciton: predictions by neural network with shape [batch size, out_features value of the last linear layer]
        :param target: ground truth, where last dim is [x, y, w, h, confidence]
        """
        prediction = prediction.reshape(-1, self.s, self.s, 5*self.b + self.c)

        # predicts looks for each cell:
        # [x, y, w, h, confidence, x, y, w, h, confidence, ..., confidences for each class]
        # where x and y - coordinates of a bounding box middle point, where (0, 0) is top left corner of a cell
        #                 and (1, 1) is a right bottom corner of a cell
        #       w and h - width and height of the bounding box, where 1 equals width or height of a cell
        #                 this values can be more than 1 (because width or height of a bounding box can be more than 
        #                 width or height of a cell) 
        #       confidence - confidence, that this bounding box contains an object
        #       confidences for each class means that this detected object belongs to class (probability map) 
        # 
        # calculates maximum IoU between anchors for each cell
        ious = torch.cat(
            [intersection_over_union(prediction[..., i:i + 4], target[..., i:i + 4]) \
                for i in range(0, 5*self.b + self.c, 5)], dim=-1)
        ious_maxes, bestbox = torch.max(ious, dim=-1, keepdim=True)
        # where ious_maxes - ious maximum value
        #       bestbox - anchor index with max IoU value

        # the last dim of the target is ground truth [x, y, w, h, confidence, confidence for each class]
        # where x and y - coordinates of grond truth bounding box if exists, else values are 0
        #       w and h - width and height of ground truth bounding box if exists, else values are 0
        #       confidence - 1 if object exists in this cell, else 0
        #       confidence for each class is one hot vector with size number of classes, where 1 means that the
        #                                 object belongs to the class and others are 0. If there are no objects
        #                                 in the cell - all values of this vector are 0.
        # binary value - object exists (value 1) or dosn't (value 0)
        # unsqueeze is necessary to keep dim of the tensor after slice
        object_exists = target[..., 4].unsqueeze(-1)
        return
