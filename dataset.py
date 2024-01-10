#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/8/23 17:29
# @Author  : ZHANG YIYI
# @File    : dataset.py
# @Software: PyCharm

import os
import pandas as pd
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset, DataLoader
ImageFile.LOAD_TRUNCATED_IMAGES = True

class MyDataset(Dataset):

    def __init__(self, imgs, labels, transform=None,target_transform=None):

        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path = self.imgs[idx]
        target = self.labels[idx]

        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class FGDataset(Dataset):

    def __init__(self, rawdata_root, anno, transform=None,target_transform=None):

        self.img_root = rawdata_root
        self.anno = anno
        self.imgs = pd.read_csv(anno, \
                           sep=" ", \
                           header=None, \
                           names=['Imagepath', 'label'])
        self.transform = transform
        self.target_transform = target_transform


    def __len__(self):
        num_lines = sum(1 for line in open(self.anno))
        return num_lines

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path = self.imgs.values[idx][0]
        path = self.img_root + '/' + path
        target = int(self.imgs.values[idx][1]) -1

        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target