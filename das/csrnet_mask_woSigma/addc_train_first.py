import sys
import os

import warnings

from model import CSRNet
from model1 import CSRNet1
from utils import save_checkpoint

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

import numpy as np
import argparse
import json
import cv2
import dataset
import time
import torch.nn.functional as F


parser = argparse.ArgumentParser(description='PyTorch CSRNet')

parser.add_argument('train_json', metavar='TRAIN',
                    help='path to train json')
parser.add_argument('test_json', metavar='TEST',
                    help='path to test json')

parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None,type=str,
                    help='path to the pretrained model')

parser.add_argument('gpu',metavar='GPU', type=str,
                    help='GPU id to use.')

parser.add_argument('task',metavar='TASK', type=str,
                    help='task id to use.')


def get_settings():
    global settings_dict
    with open("settings.json", 'r') as f:
        settings_dict = json.load(f)



def main():
    get_settings()

    global args,best_prec1, best_mse 
    
    best_prec1 = 1e6
    best_mse = 1e6
    args = parser.parse_args()
    args.original_lr = 1e-7
    args.lr = 1e-7
    args.batch_size    = 1
    args.momentum      = 0.95
    args.decay         = 5*1e-4
    args.start_epoch   = 0
    args.epochs = 400
    args.steps         = [-1,1,100,150]
    args.scales        = [1,1,1,1]
    args.workers = 0
    # args.workers = 4
    args.seed = time.time()
    args.print_freq = 400
    with open(args.train_json, 'r') as outfile:        
        train_list = json.load(outfile)
    with open(args.test_json, 'r') as outfile:       
        val_list = json.load(outfile)
        
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.cuda.manual_seed(args.seed)
    
    # model = CSRNet()
    model = CSRNet1()

    # model = model.cuda()
    
    # model1 = CSRNet1()
    # model1 = model1.cuda()

    # premodel = settings_dict['premodel_dir']

    # pre = torch.load(premodel)
    # pre = pre['state_dict']
    # model1.load_state_dict(pre)
    
    # model2 = CSRNet()
    # model2 = model2.cuda()
    # pre = torch.load(premodel)
    # pre = pre['state_dict']
    # model2.load_state_dict(pre)
    
    criterion = nn.MSELoss(size_average=False).cuda()   #深度密度图
    # count_criterion = nn.MSELoss(size_average=False).cuda() #带有限制sigma大小的密度图
    mask_criterion = nn.BCELoss(size_average=False).cuda()  #掩膜图
    
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum = args.momentum,
                                weight_decay=args.decay)

    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])

            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))
            
    for epoch in range(args.start_epoch, args.epochs):
        
        adjust_learning_rate(optimizer, epoch)
        
        # train(train_list, model, criterion,count_criterion, mask_criterion,optimizer, epoch,model1,model2)
        train(train_list, model, criterion, mask_criterion,optimizer, epoch)
        prec1,mse = validate(val_list, model, criterion, mask_criterion)
        
        
        is_best = prec1 < best_prec1
        if is_best:
            best_prec1 = prec1
            best_mse = mse

        print(' * best MAE {mae:.3f} * best MSE {mse:.3f}'.format(mae=best_prec1, mse=best_mse))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best,args.task)

    
    
def train(train_list, model, criterion, mask_criterion,optimizer, epoch):
    
    losses = AverageMeter()
    losses_d = AverageMeter()
    # losses_c = AverageMeter()
    losses_mask = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    
    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list,
                       shuffle=True,
                       transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),

                   ]), 
                       train=True, 
                       seen=model.seen,
                       batch_size=args.batch_size,
                       num_workers=args.workers),
        batch_size=args.batch_size)
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))
    
    model.train()
    end = time.time()
    
    for i,(img, target, mask_target)in enumerate(train_loader):
        # print("1")
        data_time.update(time.time() - end)
        
        img = img.cuda()
        img = Variable(img)
        
        output,mask = model(img)  # 得到图片img对应的mask   先得到掩膜
        # mask1 = torch.where(mask1>0.1,1,0)
        # output,mask2 = model2(img,mask1)  # 得到img*mask  再通过掩膜掩去背景，通过这个新的掩膜图去制作新的掩膜
        # output,mask = model(img,mask2)  # 最后用img和img*mask进行训练， output是密度图




        mask_target = Variable(mask_target)  # mask的真实图
        mask_target = mask_target.type(torch.FloatTensor).unsqueeze(0).cuda()
        target = Variable(target)  #output对应的真实图
        target = target.type(torch.FloatTensor).unsqueeze(0).cuda()
        # count_target = Variable(count_target)  # output对应的真实图
        # count_target = count_target.type(torch.FloatTensor).unsqueeze(0).cuda()

        # print(mask_target)

        # 使用torch.isnan()检测NaN值
        # nan_mask = torch.isnan(mask_target)

        # 检查是否有任何NaN值
        # has_nan = torch.any(nan_mask)

        # print(has_nan)
        # print(f"mask's dtype: {mask.dtype}，mask1's dtype: {mask1.dtype}, mask2's dtype: {mask2.dtype}, mask_target's dtype: {mask_target.dtype}")

        # loss_d = criterion(output, target) * 0.5
        loss_d = criterion(output, target)

        # test
        # mask_target = torch.clamp(mask_target, min=0., max=1.)

        mask_loss = mask_criterion(mask,mask_target) * 0.5
        
        loss = loss_d + mask_loss

        # loss_d = criterion(output, target)
        # mask_loss = mask_criterion(mask, mask_target) * 0.1
        # loss = loss_d + mask_loss

        losses.update(loss.item(), img.size(0))
        losses_d.update(loss_d.item(), img.size(0))
        losses_mask.update(mask_loss.item(), img.size(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == args.print_freq-1:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Loss_d {loss_d.val:.4f} ({loss_d.avg:.4f})\t'
                  'Loss_mask {mask_loss.val:.4f} ({mask_loss.avg:.4f})\t'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, loss_d=losses_d, mask_loss=losses_mask))
    
def validate(val_list, model, criterion, mask_criterion):
    print ('begin test')
    test_loader = torch.utils.data.DataLoader(
    dataset.listDataset(val_list,
                   shuffle=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]),  train=False),
    batch_size=args.batch_size)    
    
    model.eval()
    mae = 0
    mse = 0

    # for i, (img, target, count_target, mask_target) in enumerate(test_loader):
    for i, (img, target, mask_target) in enumerate(test_loader):
        img = img.cuda()
        img = Variable(img)

        with torch.no_grad():
            output,mask = model(img)

        # target_sum = (target.sum().type(torch.FloatTensor).cuda() + count_target.sum().type(torch.FloatTensor).cuda())/2
        target_sum = target.sum().type(torch.FloatTensor).cuda()
        mae += abs(output.data.sum() - target_sum)
        mse += (output.data.sum() - target_sum).pow(2)
        
        
    N = len(test_loader)
    mae = mae / N
    mse = torch.sqrt(mse / N)
    print('Val * MAE {mae:.3f} * MSE {mse:.3f}'
          .format(mae=mae, mse=mse))
    

    return mae,mse
        
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""    
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    
    
if __name__ == '__main__':
    main()        