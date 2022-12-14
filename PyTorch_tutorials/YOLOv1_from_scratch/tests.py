import torch
import unittest
from model import YOLOv1
from utils import intersection_over_union
from loss import YoloLoss


class TestModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = YOLOv1(
            in_channels=3, split_size=7, n_boxes=2, n_classes=20, hidden_size=4096)

    def test_darknet_output_shape(self):
        x = torch.randn(size=(8, 3, 448, 448))
        model_output = self.model.darknet(x)
        self.assertEqual(model_output.shape, torch.Size([8, 1024, 7, 7]))

    def test_model_output_shape(self):
        x = torch.randn(size=(8, 3, 448, 448))
        model_output = self.model(x)
        self.assertEqual(model_output.shape, torch.Size([8, 1470]))


class TestIoU(unittest.TestCase):

    def test_iou(self):
        # x, y, w, h
        x_pred = torch.tensor([
            [
                [0.50, 0.50, 1.00, 2.00],
                [0.50, 0.50, 2.00, 1.00],
                [0.30, 0.35, 0.40, 0.50],
            ],
            [
                [0.45, 0.40, 0.50, 2.00],
                [0.55, 0.05, 0.30, 0.30],
                [0.80, 0.85, 0.40, 0.70],
            ],
            [
                [0.60, 0.10, 0.20, 0.80],
                [0.50, 0.50, 0.60, 0.80],
                [0.60, 0.35, 0.20, 0.30],
            ],
        ])

        x_true = torch.tensor([
            [
                [0.50, 0.50, 1.00, 1.00],
                [0.50, 0.50, 1.00, 1.00],
                [0.50, 0.95, 0.20, 1.10],
            ],
            [
                [0.70, 0.90, 2.20, 0.60],
                [0.35, 0.35, 0.30, 0.50],
                [0.25, 0.40, 0.90, 0.60],
            ],
            [
                [0.50, 0.60, 1.40, 0.40],
                [0.60, 0.35, 0.20, 0.30],
                [0.50, 0.50, 0.60, 0.80],
            ],  
        ])
        ious = intersection_over_union(boxes_preds=x_pred, boxes_labels=x_true)
        self.assertTrue(torch.allclose(
            torch.tensor([
                [
                    [0.5],
                    [0.5],
                    [0.05],
                ],
                [
                    [0.1485148515],
                    [0.04347826087],
                    [0.025],
                ],
                [
                    [0.02857142857],
                    [0.125],
                    [0.125],
                ],
            ]), ious, rtol=1e-3
        ))


class TestYoloLossFunction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # lets use 5 classes and 3 anchors as example
        s = 3
        b = 3
        c = 5
        cls.loss_function = YoloLoss(s=s, b=b, c=c, lambda_coord=1)
        cls.prediction = torch.tensor([
            [       
                [  # x   , y   , w   , h   , c   , x   , y   , w   , h   , c   , x   , y   , w   , h   , c   , p   , p   , p   , p   , p
                    [0.50, 0.50, 1.00, 2.00, 0.00, 0.50, 0.50, 2.00, 1.00, 0.00, 0.30, 0.35, 0.40, 0.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                    [0.45, 0.40, 0.50, 2.00, 0.00, 0.55, 0.05, 0.30, 0.30, 0.00, 0.80, 0.85, 0.40, 0.70, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                    [0.60, 0.10, 0.20, 0.80, 0.00, 0.50, 0.50, 0.60, 0.80, 0.00, 0.60, 0.35, 0.20, 0.30, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                ],
                [
                    [0.50, 0.50, 1.00, 2.00, 0.00, 0.50, 0.50, 2.00, 1.00, 0.00, 0.30, 0.35, 0.40, 0.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                    [0.45, 0.40, 0.50, 2.00, 0.00, 0.55, 0.05, 0.30, 0.30, 0.00, 0.80, 0.85, 0.40, 0.70, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                    [0.60, 0.10, 0.20, 0.80, 0.00, 0.50, 0.50, 0.60, 0.80, 0.00, 0.60, 0.35, 0.20, 0.30, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                ],
                [
                    [0.50, 0.50, 1.00, 2.00, 0.00, 0.50, 0.50, 2.00, 1.00, 0.00, 0.30, 0.35, 0.40, 0.50, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                    [0.45, 0.40, 0.50, 2.00, 0.00, 0.55, 0.05, 0.30, 0.30, 0.00, 0.80, 0.85, 0.40, 0.70, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                    [0.60, 0.10, 0.20, 0.80, 0.00, 0.50, 0.50, 0.60, 0.80, 0.00, 0.60, 0.35, 0.20, 0.30, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                ],
            ]
        ])
        cls.target = torch.tensor([
            [
                [  # x   , y   , w   , h   , c   , p   , p   , p   , p   , p
                    [0.50, 0.50, 1.00, 1.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                    [0.70, 0.90, 2.20, 0.60, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                    [0.50, 0.60, 1.40, 0.40, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                ],
                [
                    [0.50, 0.50, 1.00, 1.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                    [0.35, 0.35, 0.30, 0.50, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                    [0.60, 0.35, 0.20, 0.30, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                ],
                [
                    [0.50, 0.95, 0.20, 1.10, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                    [0.25, 0.40, 0.90, 0.60, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                    [0.50, 0.50, 0.60, 0.80, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                ],
            ]
        ])
        ious = torch.cat(
            [intersection_over_union(cls.prediction[..., i:i + 4], cls.target[..., 0:4]) \
                for i in range(0, 5*b, 5)], dim=-1)
        _, cls.bestbox = torch.max(ious, dim=-1, keepdim=True)
    
    def test_xy_penalty(self):
        loss = self.loss_function.xy_coordinates_penalty(
            prediction=self.prediction, target=self.target, best_box_idx=self.bestbox)
        self.assertTrue(torch.allclose(loss, torch.tensor([0.27749999999999997]), rtol=1e-6))

    def test_wh_penalty(self):
        loss = self.loss_function.wh_penalty(
            prediction=self.prediction, target=self.target, best_box_idx=self.bestbox)
        self.assertTrue(torch.allclose(loss, torch.tensor([2.7383959074156614]), rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
