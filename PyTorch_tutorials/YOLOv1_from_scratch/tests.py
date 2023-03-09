import torch
import unittest


class TestModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from utils import intersection_over_union
        from model import YOLOv1

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
        from utils import intersection_over_union

        # x, y, w, h
        prediction = torch.tensor([
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

        target = torch.tensor([
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
        ious = intersection_over_union(bbox_a=prediction, bbox_b=target)
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

    def test_wh_correction_1(self):
        from utils import wh_correction


        n_anchors = 2
        n_classes = 20
        grid_size = 7
        batch_size = 8

        data = torch.randint(0, 10, size=(batch_size, grid_size, grid_size, 5*n_anchors + n_classes)) \
                    .flatten(end_dim=-2)
        wh_indexes = \
            torch.arange(3, 5*n_anchors, 5).repeat_interleave(2) + torch.arange(2).repeat(n_anchors)
        data_transformed = wh_correction(data=data, n_anchors=n_anchors, grid_size=grid_size)
        data[..., wh_indexes] = torch.pow(input=data[..., wh_indexes], exponent=2)*grid_size
        self.assertTrue(torch.equal(data, data_transformed))

    def test_wh_correction_2(self):
        from utils import wh_correction


        n_classes = 20
        grid_size = 3
        batch_size = 8

        target = torch.randint(0, 10, size=(batch_size, grid_size, grid_size, 5 + n_classes)) \
                    .flatten(end_dim=-2)
        target_transformed = wh_correction(data=target, n_anchors=1, grid_size=grid_size)
        target[..., [3, 4]] = torch.pow(input=target[..., [3, 4]], exponent=2)*grid_size
        self.assertTrue(torch.equal(target, target_transformed))        

    def test_get_best_box(self):
        from utils import get_best_box


        

if __name__ == "__main__":
    unittest.main()
