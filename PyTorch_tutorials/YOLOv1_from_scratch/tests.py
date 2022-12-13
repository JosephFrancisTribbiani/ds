import torch
import unittest
from model import YOLOv1
from utils import intersection_over_union


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
            [0.5, 0.5, 1, 2],
            [0.5, 0.5, 2, 1],
        ])
        x_true = torch.tensor([
            [0.5, 0.5, 1, 1],
            [0.5, 0.5, 1, 1],
        ])
        ious = intersection_over_union(boxes_preds=x_pred, boxes_labels=x_true)
        self.assertTrue(torch.allclose(
            torch.tensor([
                [0.5],
                [0.5],
            ]), ious, rtol=1e-3
        ))


if __name__ == "__main__":
    unittest.main()
