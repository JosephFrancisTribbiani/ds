import torch
from typing import Tuple, Union


def intersection_over_union(bbox_a: torch.tensor, bbox_b: torch.tensor, 
                            epsilon: float = 1e-6) -> torch.tensor:
    """
    Функция рассчета IoU. На вход принимает два bounding box и рассчитывает IoU между ними.
    :param bbox_a: first bounding box parameters [x, y, w, h]
                   где x и y - координаты центра bounding box
                   h и w - ширина и высота
    :param bbox_b: second bounding box parameters [x, y, w, h]
                   где x и y - координаты центра bounding box
                   h и w - ширина и высота
    :param epsilon: в случае, когда Union между двумя bounding boxes равен 0, в формуле рассчета 
                    IoU мы получаем деление на 0 (т.к. Union стоит в знаменателе). Чтобы избежать 
                    деление на 0 к Union добавляется константа epsilon, default to 1e-6
    """
    # rectangle defined by top-left and bottom-right coordinates
    a_xmin, a_xmax, a_ymin, a_ymax = \
        bbox_a[..., 0].unsqueeze(-1) - bbox_a[..., 2].unsqueeze(-1)/2, \
        bbox_a[..., 0].unsqueeze(-1) + bbox_a[..., 2].unsqueeze(-1)/2, \
        bbox_a[..., 1].unsqueeze(-1) - bbox_a[..., 3].unsqueeze(-1)/2, \
        bbox_a[..., 1].unsqueeze(-1) + bbox_a[..., 3].unsqueeze(-1)/2
    b_xmin, b_xmax, b_ymin, b_ymax = \
        bbox_b[..., 0].unsqueeze(-1) - bbox_b[..., 2].unsqueeze(-1)/2, \
        bbox_b[..., 0].unsqueeze(-1) + bbox_b[..., 2].unsqueeze(-1)/2, \
        bbox_b[..., 1].unsqueeze(-1) - bbox_b[..., 3].unsqueeze(-1)/2, \
        bbox_b[..., 1].unsqueeze(-1) + bbox_b[..., 3].unsqueeze(-1)/2
    
    # intersection of with and height as overlap x and y axises
    w_intersection, h_intersection = \
        (torch.min(a_xmax, b_xmax) - torch.max(a_xmin, b_xmin)).clamp(0), \
        (torch.min(a_ymax, b_ymax) - torch.max(a_ymin, b_ymin)).clamp(0)
    
    # intersection area
    intersection = w_intersection * h_intersection
    
    # union area, area_a is area of the rectangle bbox_a, area_b is area of the rectangle bbox_b
    area_a = bbox_a[..., 2].unsqueeze(-1)*bbox_a[..., 3].unsqueeze(-1)
    area_b = bbox_b[..., 2].unsqueeze(-1)*bbox_b[..., 3].unsqueeze(-1)
    union = area_a + area_b - intersection
    
    # calculating IoU for two rectangles bbox_a and bbox_b
    return intersection / (union + epsilon)


def wh_correction(data: torch.tensor, n_anchors: int = 1, grid_size: int = 7) -> torch.tensor:
    """
    Перед тем, как рассчитать значения IoU нам необходимо скорректировать значения w и h предсказанного
    bounding box, т.к. YOLOv1 предсказывает квадратный корень из w и h. Также, значения координат x и y 
    находятся в системе координат anchor (левый верхний угол anchor имеет координаты 0, 0), в то время как
    значения w и h являются относительными к размеру изображения. Поэтому значения w и h необходимо домножить
    на grid size, чтобы IoU был рассчитан правильно в рамках anchor.
    :param data: тензор, значения w и h которого необходимо скорректировать
    :param n_anchors: количество bounding boxes, содержащихся в тензоре data. К примеру, если data - тензор с 
                      target значениями, то значение n_anchors должно быть равно 1, т.к. каждая cell может предсказывать
                      не более одного объекта
    :param grid_size: размер сетки, делящей наш feature map на cells
    """
    return torch.where(torch.ones_like(input=data, dtype=torch.uint8) \
                            .index_fill_(-1, torch.arange(3, 5*n_anchors, 5).repeat_interleave(2) + \
                                         torch.arange(2).repeat(n_anchors), 0).to(torch.bool), 
                       data, 
                       torch.pow(data, 2)*grid_size)


def get_best_box(prediction: torch.tensor, target: torch.tensor, n_anchors: int = 2, 
                 grid_size: int = 7) -> Tuple[torch.tensor, torch.tensor]:
    """
    :param prediction: 2D tensor with shape [batch_size*grid_size*grid_size, 5*n_anchors + n_classes]
    :param taget: 2D tensor with shape [batch_size*grid_size*grid_size, 5 + n_classes]
    :param n_anchors: number of anchor boxes, default to 2
    :param grid_size: default to 7
    """
    prediction_bboxes = wh_correction(data=prediction, n_anchors=n_anchors, grid_size=grid_size)
    target_bboxes = wh_correction(data=target, n_anchors=1, grid_size=grid_size)
    ious = \
        torch.cat([intersection_over_union(bbox_a=prediction_bboxes[..., i:i + 4], 
                                           bbox_b=target_bboxes[..., 1:5]) for i in range(1, 5*n_anchors, 5)],
                  dim=-1)
    ious_maxes, bestbox = torch.max(ious, dim=-1, keepdim=True)
    return ious_maxes, bestbox


def extend_target(target: torch.tensor, bestbox: torch.tensor, bestbox_iou: torch.tensor,
                  n_anchors: int = 2) -> torch.tensor:
    """
    :param target: 2D tensor with shape [batch_size*grid_size*grid_size, 5*n_anchors + n_classes]
    :param bestbox:
    :param best_box_values:
    :param n_anchors: number of anchor boxes, default to 2
    """
    n_cells, vector_size = target.shape
    extended_target = torch.zeros(size=(n_cells, (n_anchors - 1)*5 + vector_size))
    extended_target = extended_target.scatter(-1, bestbox*5 + torch.arange(1, 5), target[..., 1:5])
    extended_target = extended_target.scatter(-1, bestbox*5, target[..., [0]]*bestbox_iou)
    extended_target[..., 5*n_anchors:] = target[..., 5:]
    return extended_target


def get_mask(size: Tuple[int], has_object: torch.tensor, bestbox: torch.tensor, n_anchors: int = 2, 
             lambda_coord: Union[int, float] = 5, lambda_noobj: Union[int, float] = 0.5) -> torch.tensor:
    """
    :param size:
    :param has_object:
    :param bestbox:
    :param n_anchors: number of anchor boxes, default to 2
    :param lambda_coord:
    :param lamda_noobj:
    """
    n_cells, vector_size = size
    mask = torch.zeros(size=(n_cells, vector_size))

    # classes penalty masking
    mask[..., 5*n_anchors:] = (1 - mask[..., 5*n_anchors:])*has_object

    # object and noobject confidence penalty masking
    confidence = \
        torch.full(size=(n_cells, n_anchors), fill_value=lambda_noobj)
    confidence = \
        confidence.where(1 - torch.zeros_like(confidence) \
                                  .scatter(-1, bestbox, has_object) \
                                  .to(torch.uint8), 
                        torch.ones_like(confidence))
    mask[..., torch.arange(0, 5*n_anchors, 5)] = confidence

    # coordinates and bounding box size penalty masking
    mask = \
        mask.scatter(-1, bestbox*5 + torch.arange(1, 5), has_object.repeat_interleave(4, -1)*lambda_coord)
    return mask
