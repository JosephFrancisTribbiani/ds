import torch
import unittest
import numpy as np
from model import YOLOv1


class TestModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = YOLOv1(
            in_channels=3, split_size=3, n_boxes=3, n_classes=80, hidden_size=496)

    def test_model_output_shape(self):
        x = torch.randn(size=(8, 3, 448, 448))
        model_output = self.model.darknet(x)
        print(model_output.shape)
        return


if __name__ == "__main__":
    unittest.main()
