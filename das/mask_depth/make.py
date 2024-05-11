import cv2
import matplotlib.pyplot as plt
import h5py
import numpy as np
from matplotlib import cm as CM
import h5py
import scipy.io as io
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy import spatial
import time
from PIL import Image,ImageFilter,ImageDraw
import time
from pylab import *
from matplotlib import cm as CM
import torch


def Smooth_heaviside(x):
    x1 = 2 - 1 / (torch.sigmoid(6000 * x) )
    x2 = torch.sigmoid(6000 * x)
    return  x2*x1


gt_path=r"G:\renqun\das\das\shanghai\part_A_final\train_data\ground_truth"
img_path=r"G:\renqun\das\das\shanghai\part_A_final\train_data\images"


filelist = os.listdir(gt_path) 
total_num = int(len(filelist)/2)
print(total_num)


for z in range(total_num):
    print(z)
    path1 = gt_path + '/IMG_' + str(z+1) +'.h5' 
    
    gt_file = h5py.File(path1,'r')

    target = np.asarray(gt_file['density'])
    target = torch.from_numpy(target)
    target=Smooth_heaviside(target)
    
    path1= gt_path.replace('ground_truth','maskh5') +'/MASK_' +str(z+1)+'.h5'
    with h5py.File(path1, 'w') as hf:
        hf['density'] = target
    z =z+1



