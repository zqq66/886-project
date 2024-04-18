import torch
import torch.nn as nn
import numpy as np


class FocalLoss(object):
    def __init__(self, gamma):
        self.gamma = gamma
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    def __call__(self, outputs, label, reduction='mean'):
        focal_weight = ((1 - outputs.sigmoid()) * label + outputs.sigmoid() * (1 - label)) ** self.gamma
        loss = self.loss_fn(outputs, label.float()) * focal_weight

        if reduction == 'mean':
            return loss.sum(-1).mean()
        elif reduction == 'sum':
            return loss.sum()
        elif reduction == 'none':
            return loss
        else:
            raise NotImplementedError