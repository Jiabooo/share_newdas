import random
import os
from PIL import Image,ImageFilter,ImageDraw
import numpy as np
import h5py
from PIL import ImageStat
import cv2
import torch
import torch.nn as nn

def Smooth_heaviside(x):
    x1 = 2 - 1 / (torch.sigmoid(1e7 * x) )
    x2 = torch.sigmoid(1e7 * x)
    return  x2*x1

def load_data(img_path,train = True):
    gt_path = img_path.replace('.jpg','.h5').replace('images','ground_truth')
    # new_gt_path = img_path.replace('.jpg','.h5').replace('images','new_gt')  # 1
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)
    # new_gt_file = h5py.File(new_gt_path)
    target = np.asarray(gt_file['density'])
    # count_target = np.asarray(new_gt_file['density'])  # 1

    target1 = torch.Tensor(target)
    mask_target = Smooth_heaviside(target1)   
    mask_target = mask_target.numpy()
    # count_target = count_target.numpy()

    if train:
        crop_size = (int(img.size[0]/2),int(img.size[1]/2))
        if random.randint(0,9)<= -1:
            
            
            dx = int(random.randint(0,1)*img.size[0]*1./2)
            dy = int(random.randint(0,1)*img.size[1]*1./2)
        else:
            dx = int(random.random()*img.size[0]*1./2)
            dy = int(random.random()*img.size[1]*1./2)
        
        
        
        img = img.crop((dx,dy,crop_size[0]+dx,crop_size[1]+dy))
        target = target[dy:crop_size[1]+dy,dx:crop_size[0]+dx]
        # count_target = count_target[dy:crop_size[1]+dy,dx:crop_size[0]+dx]
        mask_target = mask_target[dy:crop_size[1]+dy,dx:crop_size[0]+dx]
        
        
        
        if random.random()>0.8:
            target = np.fliplr(target)
            # count_target = np.fliplr(count_target)
            mask_target = np.fliplr(mask_target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    
    
    
    target = cv2.resize(target,(int(target.shape[1]/8),int(target.shape[0]/8)),interpolation = cv2.INTER_AREA)*64
    # count_target = cv2.resize(count_target, (int(count_target.shape[1] / 8), int(count_target.shape[0] / 8)), interpolation=cv2.INTER_AREA) * 64
    mask_target = cv2.resize(mask_target,(int(mask_target.shape[1]/8),int(mask_target.shape[0]/8)),interpolation = cv2.INTER_AREA)
    
    # return img, target, count_target, mask_target
    return img, target, mask_target