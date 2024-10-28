import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from image import *
import torchvision.transforms.functional as F

class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None,  train=False, seen=0, batch_size=1, num_workers=4):
        if train:
            root = root *4
        random.shuffle(root)
        
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        
    def __len__(self):
        return self.nSamples
    def __getitem__(self, index):
        assert index <= len(self), 'index range error' 
        
        img_path = self.lines[index]
        
        # img,target,count_target, mask_target,depth_target= load_data(img_path,self.train)
        img,target, mask_target,depth_target= load_data(img_path,self.train)

        
        if self.transform is not None:
            img = self.transform(img)
        # return img_path,img,target, count_target, mask_target,depth_target
        # return img_path,img,target, mask_target,depth_target
        return img,target, mask_target,depth_target

