#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/23 14:46
# @Author  : ZHANG YIYI
# @File    : generate_proto.py
# @Software: PyCharm
# -*- coding: UTF-8 -*-
from __future__ import print_function, division

import copy
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataset import MyDataset, FGDataset
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from models.ResNet_torch import resnet50
from models.eca_resnet import eca_resnet50
from collections import OrderedDict
from models.Mainmodel import DenseNet121
from util import AverageMeter, accuracy, adjust_learning_rate, is_image_file, find_classes
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

import shutil
from sklearn.model_selection import  StratifiedKFold

from torchvision import datasets, transforms

import torchvision.transforms.functional as F
import sys
from loss import EnhancedLoss


def validate(val_loader, model, args):
    """validation"""

    # switch to evaluate mode
    model.eval()
    d = {}

    with torch.no_grad():
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.item()

            # compute output
            feature, _ = model(input)

            if target in d:
                d[target].update(feature.squeeze().cpu().numpy())
            else:
                d[target] = AverageMeter()
                d[target].update(feature.squeeze().cpu().numpy())

    np.save('prototype_' + args.dataset+'_'+ args.arch + '_train.npy', d)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='leaf classification')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--batch_size", '-bs', type=int, default=1)
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                        help='model architecture: DeepGs | genonet | ResNet10 | eca_resnet18')
    parser.add_argument('--optim', type=str, default='SGD',
                        help='optim for training, Adam / SGD (default)')
    parser.add_argument('--gpu', type=str, default='0,1,2,3')
    parser.add_argument('--workers', type=int, default=8, help='workers for reading datasets')
    parser.add_argument('--num_classes', type=int, default=202)
    parser.add_argument('--pretrained', default=True, type=bool)
    parser.add_argument('--feat_dim', default=2048, type=int, help='feature dimension')
    parser.add_argument("--dataset", type=str, default="")

    args = parser.parse_args()
    print(args)

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((args.size, args.size)),
            transforms.RandomRotation(degrees=15),
            transforms.CenterCrop((448, 448)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((args.size, args.size)),
            transforms.CenterCrop((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }


    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        gpus = list(range(len(args.gpu.split(','))))
        print(gpus)
    else:
        gpus = [0]

    # load dataset


    if args.dataset == 'Soy200':
        rawdata_root = '/'
        anno_train = '/data1/SoyCultivar200/train.txt'
        anno_test = '/data1/SoyCultivar200/test.txt'
        args.num_classes = 200
    elif args.dataset == 'cotton80':
        rawdata_root = '/data1/cotton80/COTTON/images/'
        anno_train = '/data1/cotton80/COTTON/anno/train.txt'
        anno_test = '/data1/cotton80/COTTON/anno/test.txt'
        args.num_classes = 80
    elif args.dataset == 'soylocal':
        rawdata_root = '/data1/soylocal/soybean200/images/'
        anno_train = '/data1/soylocal/soybean200/anno/train.txt'
        anno_test = '/data1/soylocal/soybean200/anno/test.txt'
        args.num_classes = 200
    elif args.dataset == 'soyglo':
        rawdata_root = '/'
        anno_train = '/data1/soybeanGlo_train.txt'
        anno_test = '/data1/soybeanGlo_test.txt'
        args.num_classes = 1938
    elif args.dataset == 'soyageR1':
        rawdata_root = '/data1/R1/images/'
        anno_train = '/data1/R1/anno/train.txt'
        anno_test = '/data1/R1/anno/test.txt'
        args.num_classes = 198
    elif args.dataset == 'soyage':
        rawdata_root = '/data1/soyage/'
        anno_train = '/data1/soyage_anno/train.txt'
        anno_test = '/data1/soyage_anno/test.txt'
        args.num_classes = 198
    elif args.dataset == 'soygen':
        rawdata_root = '/data1/soygen/'
        anno_train = '/data1/soygene_anno/train.txt'
        anno_test = '/data1/soygene_anno/test.txt'
        args.num_classes = 1110


    train_dataset = FGDataset(rawdata_root, anno_train, data_transforms['val'])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                   shuffle=True, num_workers=args.workers)

    # define model
    if args.arch == 'DenseNet121':
        model = DenseNet121(args.num_classes, pretrained=args.pretrained)
    else:
        model = locals()[args.arch](pretrained=args.pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, args.num_classes)

    validate(train_dataloader, model.cuda(),args)