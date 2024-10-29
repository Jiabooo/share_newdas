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


down = 10

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
    # args.batch_size    = 4
    args.momentum      = 0.95
    args.decay         = 5*1e-4
    args.start_epoch   = 0
    args.epochs = 600
    args.steps         = [-1,1,100,150]
    args.scales        = [1,1,1,1]
    args.workers = 4
    args.seed = time.time()
    args.print_freq = 400
    with open(args.train_json, 'r') as outfile:        
        train_list = json.load(outfile)
    with open(args.test_json, 'r') as outfile:       
        val_list = json.load(outfile)
        
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.cuda.manual_seed(args.seed)
    
    model = CSRNet()
    
    model = model.cuda()
    # pre = torch.load(settings_dict["maskmodel_dir"])
    # pre = torch.load(r"D:\renqun\share_newdas\das\mask_depth2\ressultModels\A_2model_best.pth.tar")
    # pre = pre['state_dict']
    # model.load_state_dict(pre)
    
    model1 = CSRNet1()
    model1 = model1.cuda()
    pre = torch.load(settings_dict["maskmodel_dir"])
    pre = pre['state_dict']
    model1.load_state_dict(pre)
    
    criterion = nn.MSELoss(size_average=False).cuda() #
    mask_criterion = nn.BCELoss(size_average=False).cuda()
    
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
        
        train(train_list, model, criterion,mask_criterion,optimizer, epoch,model1)
        prec1,mse = validate(val_list, model, criterion,mask_criterion,model1)
        
        
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

    
    
def train(train_list, model, criterion,mask_criterion,optimizer, epoch,model1):
    
    losses = AverageMeter()
    losses_d = AverageMeter()
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
    model1.eval()
    end = time.time()
    
    for i,(img, target,mask_target,depth)in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        img = img.cuda()
        img = Variable(img)
        
        output1,mask1 = model1(img) 
        # output1 = output1/down
        mask1 = torch.where(mask1>0.01,1,0)
        output1 = torch.where(output1>0.01,1,0)
        depth = depth.type(torch.FloatTensor).unsqueeze(0).cuda()*output1
        

        output,mask = model(img,depth,mask1)
        mask_target = Variable(mask_target)
        target = Variable(target)
        target = target.type(torch.FloatTensor).unsqueeze(0).cuda()
        mask_target = mask_target.type(torch.FloatTensor).unsqueeze(0).cuda()

        # test
        mask_target = torch.clamp(mask_target, min=0.0, max=1.0)

        # print("mask1 min/max:", mask1.min().item(), mask1.max().item())
        # print("output1 min/max:", output1.min().item(), output1.max().item())
        # print("mask min/max:", mask.min().item(), mask.max().item())
        # print("mask_target min/max:", mask_target.min().item(), mask_target.max().item())

        loss_d = criterion(output, target)
        mask_loss = mask_criterion(mask,mask_target)*0.1
        
        loss = loss_d + mask_loss
        
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
    
def validate(val_list, model, criterion, mask_criterion,model1):
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
    model1.eval()
    mae = 0
    mse = 0

    for i, (img, target,mask_target,depth) in enumerate(test_loader):
        img = img.cuda()
        img = Variable(img)

        
        with torch.no_grad():
            output1,mask1 = model1(img)
            # output1 = output1/down
            mask1 = torch.where(mask1>0.01,1,0)
            output1 = torch.where(output1>0.01,1,0)
            depth = depth.type(torch.FloatTensor).unsqueeze(0).cuda()*output1
            
            output,mask = model(img,depth,mask1)
        # mae += abs(output.data.sum()/down - (target.sum()/down).type(torch.FloatTensor).cuda())
        # mse += (output.data.sum()/down - (target.sum()/down).type(torch.FloatTensor).cuda()).pow(2)
        mae += abs(output.data.sum() - (target.sum()).type(torch.FloatTensor).cuda())
        mse += (output.data.sum() - (target.sum()).type(torch.FloatTensor).cuda()).pow(2)
        
        
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