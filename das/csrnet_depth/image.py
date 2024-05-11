import random
import os
from PIL import Image,ImageFilter,ImageDraw
import numpy as np
import h5py
from PIL import ImageStat
import cv2
import torch
import torch.nn as nn
import glob
import torch
import utils
from matplotlib import cm as CM
from torchvision.transforms import Compose
from midas.dpt_depth import DPTDepthModel
from midas.midas_net import MidasNet
from midas.midas_net_custom import MidasNet_small
from midas.transforms import Resize, NormalizeImage, PrepareForNet
from matplotlib import pyplot as plt

def Smooth_heaviside(x):
    x1 = 2 - 1 / (torch.sigmoid(1e7 * x) )
    x2 = torch.sigmoid(1e7 * x)
    return  x2*x1
    
def load_data(img_path,train = True):
    gt_path = img_path.replace('.jpg','.h5').replace('images','ground_truth')
    depth_path = img_path.replace('.jpg','.h5').replace('images','depth_map')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])
    depth_file = h5py.File(depth_path)
    depth_target = np.asarray(depth_file['depth'])
    depth_target = (depth_target-np.min(depth_target))/(np.max(depth_target)-np.min(depth_target))
    
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
        depth_target = depth_target[dy:crop_size[1]+dy,dx:crop_size[0]+dx]
        
        
        
        if random.random()>0.8:
            target = np.fliplr(target)
            depth_target = np.fliplr(depth_target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    
    target = cv2.resize(target,(int(target.shape[1]/8),int(target.shape[0]/8)),interpolation = cv2.INTER_LINEAR)*64
    depth_target = cv2.resize(np.float32(depth_target),(int(depth_target.shape[1]/8),int(depth_target.shape[0]/8)),interpolation = cv2.INTER_LINEAR)
    
    return img,target, depth_target