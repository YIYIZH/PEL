#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/13 17:07
# @Author  : ZHANG YIYI
# @File    : memory.py
# @Software: PyCharm

import torch
from torch import nn
import math
import numpy as np
from util import AverageMeter

class PrototypeMemory(nn.Module):
    """
    memory buffer that stores prototypes of each category.
    """
    def __init__(self, featSize, classSize, proto_path, momentum=0.5):
        super(PrototypeMemory, self).__init__()
        self.momentum = momentum
        self.memory = torch.rand(classSize, featSize)
        proto = np.load(proto_path, allow_pickle=True).item()
   
        for k in proto:
            norm = np.power(np.sum(np.power(proto[k].avg,2), axis=0), 0.5)
            self.memory[k] = torch.from_numpy(np.divide(proto[k].avg, norm))

        self.memory = self.memory.cuda()


    def forward(self, f,  y):
        f_norm = f.pow(2).sum(1, keepdim=True).pow(0.5)
        f = f.div(f_norm)
        out_f = torch.mm(f, self.memory.transpose(0,1))

        feat = {}
        y_unique = []
        feat_unique = []

        with torch.no_grad():
            # update memory
            for i in range(len(y)):
                cls = y[i].item()
                if cls not in feat:
                    feat[cls] = AverageMeter()
                    feat[cls].update(f[i])
                else:
                    feat[cls].update(f[i])
            for key in feat:
                y_unique.append(key)
                feat_unique.append(feat[key].avg)
            y_unique = torch.tensor(y_unique).cuda()
            feat_unique = torch.stack(feat_unique, 0).cuda()

            l_pos = torch.index_select(self.memory, 0, y_unique)
            l_pos = l_pos.mul(self.momentum)
            l_pos = l_pos.add(torch.mul(feat_unique, 1 - self.momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            l_pos = l_pos.div(l_norm)
            self.memory = self.memory.index_copy(0, y_unique, l_pos)

        return out_f, self.memory

