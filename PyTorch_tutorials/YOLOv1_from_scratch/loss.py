import torch
import torch.nn as nn
from typing import Union
from utils import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self, s: int = 7, b: int = 2, c: int = 20, lambda_noobj: Union[float, int] = 0.5, 
                 lambda_coord: Union[int, float] = 5) -> None:
        """
        :param s: split size (number of cells is s**2)
        :param b: numeber of anchors per cell
        :param c: number of classes
        """
        super().__init__()
        self.s = s
        self.b = b
        self.c = c
        self.lambda_noobj = lambda_noobj
        self.lambda_coord = lambda_coord

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
            [intersection_over_union(prediction[..., i:i + 4], target[..., 0:4]) \
                for i in range(0, 5*self.b, 5)], dim=-1)
        ious_maxes, bestbox = torch.max(ious, dim=-1, keepdim=True)
        # where ious_maxes - ious maximum value
        #       bestbox - anchor index with max IoU value

        # loss function of the YOLO consists of several parts
        # Part 1: x and y penalty
        # penalize only an anchor box which responsible for an object detection (has maximum IoU value) 
        # and if object exists in this cell
        xy_loss = self.xy_coordinates_penalty(prediction=prediction, target=target, best_box_idx=bestbox)

        # Part 2: w and h penalty
        # like in part 1, it penalize only an anchor box which responsible for an object detection (has maximum IoU value) 
        # and if object exists in this cell
        wh_loss = self.wh_coordinates_penalty(prediction=prediction, target=target, best_box_idx=bestbox)

        # Part 3:


        # Part 4:


        # Part 5:

        return xy_loss + wh_loss

    def xy_coordinates_penalty(self, prediction: torch.tensor, target: torch.tensor, best_box_idx: torch.tensor) -> torch.tensor:
        """
        Calculates the first part of the loss function, where we penalize for x and y coordinates deviation from ground truth.
        :param prediction: torch tensor with model predictions with size [batch size, split size, split size, 5*num_anchors + num_classes]
        :param target: tesor with target values with shape [batch_size, split_size, split size, 5 + num_classes]
        :param best_box_idx: torch tensor with index of anchor box which has maximum IoU with ground truth (size [batch size, split size, split size, 1]).
                             Value 1 in this tensor for example means that second anchor box has hieghtest IoU value with ground truth. Other anchor boxes are not
                             taken into account for loss function calculation.
        """
        # extract x and y coordinates of best anchor boxes (which have highest IoU values)
        xy_of_best_anchor = torch.gather(input=prediction, dim=-1, index=best_box_idx*5 + torch.arange(2))

        # the last dim of the target is ground truth [x, y, w, h, confidence, confidence for each class]
        # where x and y - coordinates of ground truth bounding box if exists, else values are 0
        #       w and h - width and height of ground truth bounding box if exists, else values are 0
        #       confidence - 1 if object exists in this cell, else 0
        #       confidence for each class is one hot vector with size number of classes, where 1 means that the
        #                                 object belongs to the class and others are 0. If there are no objects
        #                                 in the cell - all values of this vector are 0.
        # binary value - object exists (value 1) or dosn't (value 0)
        # unsqueeze is necessary to keep dim of the tensor after slice
        object_exists = target[..., 4].unsqueeze(-1)

        # calculate part of the YOLO loss function which is responcible for x and y coordinates
        return torch.sum(object_exists * torch.pow(xy_of_best_anchor - target[..., :2], exponent=2)) * self.lambda_coord


    def wh_coordinates_penalty(self, prediction: torch.tensor, target: torch.tensor, best_box_idx: torch.tensor) -> torch.tensor:
        """
        Calculates the second part of the loss function, where we penalize for w and h values deviation from ground truth.
        :param prediction: torch tensor with model predictions with size [batch size, split size, split size, 5*num_anchors + num_classes]
        :param target: tesor with target values with shape [batch_size, split_size, split size, 5 + num_classes]
        :param best_box_idx: torch tensor with index of anchor box which has maximum IoU with ground truth (size [batch size, split size, split size, 1]).
                             Value 1 in this tensor for example means that second anchor box has hieghtest IoU value with ground truth. Other anchor boxes are not
                             taken into account for loss function calculation.
        """
        # extract x and y coordinates of best anchor boxes (which have highest IoU values)
        wh_of_best_anchor = torch.gather(input=prediction, dim=-1, index=best_box_idx*5 + torch.arange(2) + 2)
        object_exists = target[..., 4].unsqueeze(-1)

        # calculate part of the YOLO loss function which is responcible for x and y coordinates
        return torch.sum(object_exists * torch.pow(wh_of_best_anchor - target[..., 2:4], exponent=2)) * self.lambda_coord
