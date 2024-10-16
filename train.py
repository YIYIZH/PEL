
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/13 14:11
# @Author  : ZHANG YIYI
# @File    : train.py
# @Software: PyCharm

from __future__ import print_function, division

import copy
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataset import FGDataset
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from models.ResNet_torch import resnet50, resnet18
from collections import OrderedDict
from util import AverageMeter, accuracy, adjust_learning_rate, is_image_file, find_classes
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.Mainmodel import DenseNet121

import shutil
from torchvision import datasets, transforms

import torchvision.transforms.functional as F
import sys
from loss import EnhancedLoss

def train(train_dataloader, model, criterion, args, epoch):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_pel = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_dataloader):
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        feature, output = model(input)
        loss = criterion(output, feature, target)
        loss_pel.update(loss.item(), input.size(0))

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if idx % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss_pel.val:.4f} ({loss_pel.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                idx, len(train_dataloader), batch_time=batch_time, loss=loss_pel,
                top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, loss_pel.avg

def validate(val_loader, model, criterion, args):
    """validation"""
    batch_time = AverageMeter()
    loss_pel = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            
            feature, output = model(input)
            loss = criterion(output, feature, target)
            loss_pel.update(loss.item(), input.size(0))

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss_pel {loss_pel.val:.4f} ({loss_pel.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss_pel=loss_pel,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, loss_pel.avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='leaf classification')
    parser.add_argument('--print_freq', type=int, default=30, help='print frequency')
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--batch_size", '-bs', type=int, default=8)
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                        help='model architecture: DeepGs | DenseNet121 | ResNet10 | eca_resnet18')
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument('--lr_decay_epochs', type=str, default='60,80,100', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', '-ldr', type=float, default=0.5, help='decay rate for learning rate')
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument('--optim', type=str, default='SGD',
                        help='optim for training, Adam / SGD (default)')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--workers', type=int, default=4, help='workers for reading datasets')
    parser.add_argument('--num_classes', type=int, default=80)
    parser.add_argument("--save", type=str, default=False)
    parser.add_argument("--savepath", type=str, default="./weights/")
    parser.add_argument('--pretrained', default=True, type=bool)
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--proto_path', type=str, default='')
    parser.add_argument('--feat_dim', default=2048, type=int, help='feature dimension')
    parser.add_argument('--edl_t', default=1, type=float, help='temperature parameter for kd loss')
    parser.add_argument('--edl_r', default=6, type=float, help='weights of the original label beta')
    parser.add_argument('--edl_m', default=0.9, type=float, help='momentum for memory updates')
    parser.add_argument('--sim_t', default=1, type=float, help='temperature parameter t2 for softmax')
    parser.add_argument("--dataset", type=str, default="soyglo")

    args = parser.parse_args()
    print(args)

    # Data augmentation and normalization for training
    # Data normalization for validation
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

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))

    #load dataset
    if args.dataset == 'cotton80':
        rawdata_root = '/data1/zyy/cotton80/COTTON/images/'
        anno_train = '/data1/zyy/cotton80/COTTON/anno/train.txt'
        anno_test = '/data1/zyy/cotton80/COTTON/anno/test.txt'
        args.num_classes = 80
    elif args.dataset == 'soylocal':
        rawdata_root = '/data1/zyy/soylocal/soybean200/images/'
        anno_train = '/data1/zyy/soylocal/soybean200/anno/train.txt'
        anno_test = '/data1/zyy/soylocal/soybean200/anno/test.txt'
        args.num_classes = 200
    elif args.dataset == 'soyglo':
        rawdata_root = '/'
        anno_train = '/data1/zyy/soybeanGlo_train.txt'
        anno_test = '/data1/zyy/soybeanGlo_test.txt'
        args.num_classes = 1938
    elif args.dataset == 'soyageR1':
        rawdata_root = '/data1/R1/images/'
        anno_train = '/data1/R1/anno/train.txt'
        anno_test = '/data1/R1/anno/test.txt'
        args.num_classes = 198
    elif args.dataset == 'soyage':
        rawdata_root = '/data1/zyy/soyage/'
        anno_train = '/data1/zyy/soyage_anno/train.txt'
        anno_test = '/data1/zyy/soyage_anno/test.txt'
        args.num_classes = 198
    elif args.dataset == 'soygen':
        rawdata_root = '/data1/soygen/'
        anno_train = '/data1/soygene_anno/train.txt'
        anno_test = '/data1/soygene_anno/test.txt'
        args.num_classes = 1110
    elif args.dataset == 'soycultivar200':
        rawdata_root = '/'
        anno_train = '/data1/SoyCultivar200/train.txt'
        anno_test = '/data1/SoyCultivar200/test.txt'
        args.num_classes = 200

    train_dataset = FGDataset(rawdata_root, anno_train, data_transforms['train'])
    val_dataset = FGDataset(rawdata_root, anno_test, data_transforms['val'])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                  shuffle=True, num_workers=args.workers)
    test_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                                  shuffle=False, num_workers=args.workers)

    # define model
    if args.arch == 'DenseNet121':
        model = DenseNet121(args.num_classes, pretrained=args.pretrained) #weights=DenseNet121_Weights.DEFAULT
        args.feat_dim = 1024
    elif args.arch == 'resnet50':
        model = locals()[args.arch](pretrained=args.pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, args.num_classes)
        args.feat_dim = 2048
    if args.resume:
        checkpoint = torch.load(args.model_path)
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model'].items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

    trainable_list = nn.ModuleList([])
    trainable_list.append(model)

    # define criterion
    criterion = EnhancedLoss(args)
    trainable_list.append(criterion.embed)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
        criterion.cuda()

    # Observe that all parameters are being optimized.
    if args.optim=='SGD':
        optimizer = optim.SGD(trainable_list.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    elif args.optim=='Adam':
        optimizer = optim.Adam(trainable_list.parameters(),
                               lr=args.lr,
                               betas=(0.9, 0.999),
                               eps=1e-08,
                               weight_decay=args.weight_decay)


    best_acc = 0
    for epoch in range(args.epoch):
        adjust_learning_rate(epoch, args, optimizer)
        print("==> training ")
        time1 = time.time()
        train_acc, train_loss = train(train_dataloader, model, criterion, args, epoch)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        test_acc, test_acc_top5, test_loss = validate(test_dataloader, model, criterion, args)
        #save the best model
        if  test_acc > best_acc:
            best_acc = test_acc
            if args.save:
                state = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'best_acc': best_acc,
                }
                save_file = os.path.join(args.savepath, '{}_loss{}_data{}_best.pth'.format(args.arch,args.loss,args.dataset))
                print('saving the best model!')
                torch.save(state, save_file)
        print('best accuracy:', best_acc)






