import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average=False)
        
    def forward(self, inputs, target):
        n, c, h, w = inputs.size()
        # log_p: (n, c, h, w)
        log_p = F.log_softmax(inputs, dim=1)
        # log_p: (n*h*w, c)
        log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
        log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
        log_p = log_p.view(-1, c)
        # target: (n*h*w,)
        mask = target >= 0
        target = target[mask]
        loss = self.nll_loss(log_p, target)
        loss /= mask.data.sum()
        #return self.nll_loss(F.log_softmax(inputs, dim=1), targets)
        return loss
                         
def torch_mean_iou(pred, labels):
    '''
    Compute mean IoU score over 6 classes
    '''
    mean_iou = 0
    for i in range(6):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        if (tp_fp + tp_fn - tp) == 0:
            iou = 0
        else:
            iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou
    return mean_iou / 6.0