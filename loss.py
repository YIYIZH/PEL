#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/14 10:29
# @Author  : ZHANG YIYI
# @File    : loss.py
# @Software: PyCharm

import torch
from torch import nn
from memory import PrototypeMemory
import torch.nn.functional as F

eps = 1e-7

class EnhancedLoss(nn.Module):
    def __init__(self, opt):
        super(EnhancedLoss, self).__init__()
        self.embed = ECA(opt.num_classes)
        self.prototype = PrototypeMemory(opt.feat_dim, opt.num_classes, opt.proto_path, opt.edl_m)
        self.criterion = KdLoss(opt.edl_t)
        self.r = opt.edl_r
        self.t = opt.sim_t

    def forward(self, output, feature, label):
        out_f, _ = self.prototype(feature, label)
        out_f = F.softmax(out_f / self.t, dim=1)
        ones = torch.sparse.torch.eye(output.size(1)).cuda()
        label1 = ones.index_select(0, label)
        out_f = self.r * label1 + out_f
        loss = self.criterion(output, out_f)
        return loss

class KdLoss(nn.Module):
    def __init__(self, T):
        super(KdLoss, self).__init__()
        self.T = T

    def forward(self, y_predict, y_label):
        y_predict = F.log_softmax(y_predict/self.T, dim=1)
        y_label = F.softmax(y_label/self.T, dim=1)
        loss = F.kl_div(y_predict, y_label, reduction='sum') * (self.T ** 2) / y_predict.shape[0]
        return loss

class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in, dim_out):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x

class ECA(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Two different branches of ECA module
        y = self.conv(x.unsqueeze(-2)).squeeze(-2)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        return x * y


