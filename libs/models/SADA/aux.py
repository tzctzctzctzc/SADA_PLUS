import torch

def check_overlap_gt_prediction_basic(pts, gt_offsets, pred_offsets):
    
    # Decode the predictions to start-end
    pred_seg_left = pts[:, 0] - pred_offsets[:, 0] * pts[:, 3]
    pred_seg_right = pts[:, 0] + pred_offsets[:, 1] * pts[:, 3]
    
    # Decode the GT to start-end
    gt_seg_left = pts[:, 0] - gt_offsets[:, 0] * pts[:, 3]
    gt_seg_right = pts[:, 0] + gt_offsets[:, 1] * pts[:, 3]
    
    # Check if there is overlap (mask that indicates wheathe the predicted segment overlaps with the GT segment even if just a bit)
    overlap = (pred_seg_left <= gt_seg_right) & (pred_seg_right >= gt_seg_left) & (pred_seg_left < pred_seg_right) # shape: T'
    
    return overlap


def check_overlap_gt_prediction(pts, gt_offsets, pred_offsets, iou_threshold=0.2):
    # Discard if it does not overlap with an action given the IOU threshold
    
    # Decode the predictions to start-end
    pred_seg_left = pts[:, 0] - pred_offsets[:, 0] * pts[:, 3]  # shape: B(T_i)F
    pred_seg_right = pts[:, 0] + pred_offsets[:, 1] * pts[:, 3] # shape: B(T_i)F
    
    # Decode the GT to start-end
    gt_seg_left = pts[:, 0] - gt_offsets[:, 0] * pts[:, 3]  # shape: B(T_i)F    
    gt_seg_right = pts[:, 0] + gt_offsets[:, 1] * pts[:, 3] # shape: B(T_i)F
    
    intersection_left = torch.max(pred_seg_left.unsqueeze(1), gt_seg_left.unsqueeze(0)) # shape: B(T_i)F x nGT
    intersection_right = torch.min(pred_seg_right.unsqueeze(1), gt_seg_right.unsqueeze(0)) # shape: B(T_i)F x nGT
    intersection = torch.clamp(intersection_right - intersection_left, min=0) # shape: B(T_i)F x nGT
    union = (pred_seg_right.unsqueeze(1) - pred_seg_left.unsqueeze(1)) + (gt_seg_right.unsqueeze(0) - gt_seg_left.unsqueeze(0)) - intersection # shape: B(T_i)F x nGT
    iou = intersection / (union + 1e-5)  # Adding a small epsilon to avoid division by zero

    iou_max, _ = torch.max(iou, dim=1) # shape: B(T_i)F

    # Check if there is at least one IoU greater than the threshold
    overlap = iou_max >= iou_threshold
    
    return overlap



def compute_iou(pts, gt_offsets, pred_offsets, iou_threshold=0.2):
    # Discard if it does not overlap with an action given the IOU threshold
    
    # Decode the predictions to start-end
    pred_seg_left = pts[:, 0] - pred_offsets[:, 0] * pts[:, 3]  # shape: B(T_i)F
    pred_seg_right = pts[:, 0] + pred_offsets[:, 1] * pts[:, 3] # shape: B(T_i)F
    
    # Decode the GT to start-end
    gt_seg_left = pts[:, 0] - gt_offsets[:, 0] * pts[:, 3]  # shape: B(T_i)F    
    gt_seg_right = pts[:, 0] + gt_offsets[:, 1] * pts[:, 3] # shape: B(T_i)F
    
    intersection_left = torch.max(pred_seg_left.unsqueeze(1), gt_seg_left.unsqueeze(0)) # shape: B(T_i)F x nGT
    intersection_right = torch.min(pred_seg_right.unsqueeze(1), gt_seg_right.unsqueeze(0)) # shape: B(T_i)F x nGT
    intersection = torch.clamp(intersection_right - intersection_left, min=0) # shape: B(T_i)F x nGT
    union = (pred_seg_right.unsqueeze(1) - pred_seg_left.unsqueeze(1)) + (gt_seg_right.unsqueeze(0) - gt_seg_left.unsqueeze(0)) - intersection # shape: B(T_i)F x nGT
    iou = intersection / (union + 1e-5)  # Adding a small epsilon to avoid division by zero

    iou_max, _ = torch.max(iou, dim=1) # shape: B(T_i)F

    return iou_max


