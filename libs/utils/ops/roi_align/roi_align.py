# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from . import Align1D as _align_1d
#import Align1D as _align_1d


class _Align1D(Function):
    @staticmethod
    def forward(ctx, input, roi, feature_dim, ratio):
        ctx.save_for_backward(roi)
        ctx.feature_dim = feature_dim
        ctx.input_shape = input.size()
        ctx.sampling_ratio = ratio
        output = _align_1d.forward(
            input, roi, feature_dim, ratio
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        rois, = ctx.saved_tensors
        feature_dim = ctx.feature_dim
        bs, ch, t = ctx.input_shape
        ratio = ctx.sampling_ratio
        grad_input = _align_1d.backward(
            grad_output,
            rois,
            feature_dim,
            bs,
            ch,
            t,
            ratio
        )
        return grad_input, None, None, None, None

align1d = _Align1D.apply

class ROIAlign(nn.Module):
    def __init__(self, feature_dim, ratio=0):
        super(ROIAlign, self).__init__()
        self.feature_dim = feature_dim
        self.ratio = ratio

    def forward(self, input, rois):
        # print('- input shape is', input.shape)
        # print('- input mean is', input.mean())
        # print('- rois shape is', rois.shape)
        # print('- rois is on', rois.get_device())
        assert input.device == rois.device, 'Align operation requires ' + \
                                            'both feature and roi are on the same device! ' + \
                                            'Get feature on {} but roi on {}'.format(input.device, rois.device)

        out = align1d(input, rois, self.feature_dim, self.ratio)
        # print('- output shape is', out.shape)
        # print('- output mean is', out.mean())
        return out

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "feature_dim=" + str(self.feature_dim)
        tmpstr += "sampling_ratio=" + str(self.ratio)
        tmpstr += ")"
        return tmpstr




# ---------- Debugging ----------
'''
input = torch.tensor([[[1.,2,3,4,5,6,7,8,9,10],
                       [11,12,13,14,15,16,17,18,19,20],
                       [21,22,23,24,25,26,27,28,29,30]]]).cuda()    # B x T x F
proposal = torch.tensor([[0.,0.,1.],[0.,1.,2.]]).cuda()   # B x 3

print("Testing ROIAlign:")
from roi_align import ROIAlign
import einops
input = einops.rearrange(input, 'b t f -> b f t')
layer = ROIAlign(16, 0).cuda(3)
output = layer(input, proposal)
print("output has shape {}, with mean {}".format(output.shape, torch.mean(output)))
print(output)


print("Testing ROIPooling:")
import torch.nn.functional as F
new_segm = proposal[:, 1:].clone()
new_segm = torch.clamp(new_segm, min=0, max=input.shape[1])

def roi_pooling(input, new_segm):
    print(new_segm)
    
    roi_features = input[0, new_segm[0]:new_segm[1] + 1]  # shape: T' x F
    
    # max pool to fixed size
    roi_features = roi_features.unsqueeze(0).permute(0, 2, 1)  # shape: 1 x F x T'
    roi_features = F.adaptive_max_pool1d(roi_features, 10)
    roi_features = roi_features.squeeze(0).permute(1, 0)  # shape: T' x F
    return roi_features
    
roi_features = list()
rois = torch.stack([roi_pooling(input, seg) for seg in new_segm.int()], dim=0)  # shape: num_segs x pool_res x F
roi_features.append(rois)
roi_features = torch.stack(roi_features, dim=0)    # shape: BS x num_segs x pool_res x F
    

print("output has shape {}, with mean {}".format(roi_features.shape, torch.mean(roi_features)))
print(roi_features)
'''