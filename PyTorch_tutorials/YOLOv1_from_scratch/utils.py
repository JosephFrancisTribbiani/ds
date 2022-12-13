import torch


def intersection_over_union(boxes_preds: torch.tensor, boxes_labels: torch.tensor, 
                            epsilon: float = 1e-6) -> torch.tensor:
    """
    :param boxes_preds: tensor with bounding boxes, where the last dimension represents x, y, w, h coordinates
                        of bounding box
    :param boxes_labels:
    """
    # first of all we have to calculate coordinates of bounding box corners coordinates
    # x_left = x_center - width / 2
    # x_right = x_center + width / 2
    # y_top = y_center - height / 2
    # y_bottom = y_center + height / 2
    #
    # prediction bounding box
    bbox_pred_x_left   = boxes_preds[..., 0].unsqueeze(-1) - boxes_preds[..., 2].unsqueeze(-1) / 2
    bbox_pred_x_right  = boxes_preds[..., 0].unsqueeze(-1) + boxes_preds[..., 2].unsqueeze(-1) / 2
    bbox_pred_y_top    = boxes_preds[..., 1].unsqueeze(-1) - boxes_preds[..., 3].unsqueeze(-1) / 2
    bbox_pred_y_bottom = boxes_preds[..., 1].unsqueeze(-1) + boxes_preds[..., 3].unsqueeze(-1) / 2

    # true bounding box
    bbox_true_x_left   = boxes_labels[..., 0].unsqueeze(-1) - boxes_labels[..., 2].unsqueeze(-1) / 2
    bbox_true_x_right  = boxes_labels[..., 0].unsqueeze(-1) + boxes_labels[..., 2].unsqueeze(-1) / 2
    bbox_true_y_top    = boxes_labels[..., 1].unsqueeze(-1) - boxes_labels[..., 3].unsqueeze(-1) / 2
    bbox_true_y_bottom = boxes_labels[..., 1].unsqueeze(-1) + boxes_labels[..., 3].unsqueeze(-1) / 2

    # intersection area calculation
    x_left   = torch.max(bbox_pred_x_left,   bbox_true_x_left)
    x_right  = torch.min(bbox_pred_x_right,  bbox_true_x_right)
    y_top    = torch.max(bbox_pred_y_top,    bbox_true_y_top)
    y_bottom = torch.min(bbox_pred_y_bottom, bbox_true_y_bottom)

    # .clamp(0) is for the case when they do not intersect
    intersection_area = (x_right - x_left).clamp(0) * (y_bottom - y_top).clamp(0)

    # bounding boxes areas calculation
    bbox_pred_area = torch.abs((bbox_pred_x_right - bbox_pred_x_left) * (bbox_pred_y_bottom - bbox_pred_y_top))
    bbox_true_area = torch.abs((bbox_true_x_right - bbox_true_x_left) * (bbox_true_y_bottom - bbox_true_y_top))
    
    return intersection_area / (bbox_pred_area + bbox_true_area - intersection_area + epsilon)
