# Mostly copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

'''Focal loss implementation'''

import torch
import torch.nn.functional as F
from einops import rearrange

class CategoricalSoftmaxFocalLoss(torch.nn.Module):
    "Focal loss implemented using F.cross_entropy"
    def __init__(self, alpha:float = 1.0, gamma: float = 2.0, weight=None, reduction: str = 'mean') -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = torch.tensor([1,1,1,1,1,1,1,1,0.1])
        self.reduction = reduction

    def forward(self, inp: torch.Tensor, targ: torch.Tensor, num_boxes):
        # inp: shape BxQxC
        # targ: shape: BxQ
        b, q, c = inp.shape

        inp = rearrange(inp, 'b q c -> (b q) c')
        targ = rearrange(targ, 'b q -> (b q)')

        ce_loss = torch.nn.functional.cross_entropy(inp, targ, reduction='none', weight=self.weight.to(inp.device))  # important to add reduction='none' to keep per-batch-item loss
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)  # mean over all the values

        #focal_loss = rearrange(focal_loss, '(b q) -> b q', b = b)

        # Average over the batch size
        focal_loss = torch.mean(focal_loss, 0)

        # Sum over the anchors and normalize with the true number of segments
        return focal_loss.sum() / num_boxes


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """

    # inputs: shape BxQxC
    # targets: shape: BxQxC
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    prob = inputs.sigmoid()
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # This averages over all the predictions Q, sums all of it (the batch dimension), and averages over the number of true intervals
    #return loss.mean(1).sum() / num_boxes
    return loss.mean(1).sum() / num_boxes    # TODO: Change this!