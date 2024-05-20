import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
import scipy
import json
import torchvision.transforms.functional as F
from matplotlib import cm as CM
from image import *
from model import CSRNet
from model1 import CSRNet1
import torch
from matplotlib import cm as c
from torchvision import datasets, transforms
from utils import cal_para
import dataset
from torch.autograd import Variable

transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])
def main():
    

    
    with open("A_test.json", 'r') as outfile:
        test_list = json.load(outfile)
        
        
        
    model = CSRNet()
    pretrained=torch.load(r"D:\renqun\share_newdas\das\mask_depth2\0model_best.pth.tar")
    model = model.cuda()
    model.load_state_dict(pretrained['state_dict'])
    
    
    mask_model = CSRNet1()
    pretrained=torch.load(r"D:\renqun\share_newdas\das\mask_depth2\0model_best.pth.tar")
    mask_model = mask_model.cuda()
    mask_model.load_state_dict(pretrained['state_dict'])
    
    test(test_list, model,mask_model)


def test(test_list, model,mask_model):
    print('begin test')
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(test_list,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),
                            ]),
                            train=False),
        shuffle=False,
        batch_size=1)

    model.eval()
    mae = 0
    mse = 0
    mae1 = 0
    mse1 = 0
    all = 0
    for i, (img_path,img, target,mask_target,depth) in enumerate(test_loader):
        img1 = img_path[0]
        img1 = Image.open(img1).convert('RGB')
        img1 = np.asarray(img1)
        
        img = img.cuda()
        img = Variable(img)
        with torch.no_grad():
            output1,mask = mask_model(img)
            mask = torch.where(mask>0.1,1,1)
            output1 = torch.where(output1>0.1,0,0)

            depth = depth.type(torch.FloatTensor).unsqueeze(0).cuda()*output1
            #output,mask = model(img,depth, mask)
            output,mask = model(img,mask,depth)
            mask1 = mask.clone().detach().cuda()
            
            mae += abs(output.data.sum() - target.sum().type(torch.FloatTensor).cuda())
            mse += (output.data.sum() - target.sum().type(torch.FloatTensor).cuda()).pow(2)
            all += (output.data.sum() - target.sum().type(torch.FloatTensor).cuda())
            
            output = np.asarray(output.detach().cpu().reshape(output.detach().cpu().shape[2],output.detach().cpu().shape[3]))
            mask = np.asarray(mask.detach().cpu().reshape(mask.detach().cpu().shape[2],mask.detach().cpu().shape[3]))
            mask = cv2.resize(np.float32(mask),(img.shape[3],img.shape[2]),interpolation = cv2.INTER_AREA)
            mask = np.where(mask>=0.005,1,0)
            #mask = cv2.blur(mask, (15,15))
            #mask = np.where(mask>=0.001,1,0)
            l = img1.copy()
            for i in range(img1.shape[2]):
                l[:,:,i] = img1[:,:,i]*mask

            l = transform(l)
            output,mask = model(l.unsqueeze(0).cuda(),mask1,depth)
            mae1 += abs(output.data.sum() - target.sum().type(torch.FloatTensor).cuda())
            mse1 += (output.data.sum() - target.sum().type(torch.FloatTensor).cuda()).pow(2)
            
    N = len(test_loader)
    mae = mae / N
    mse = torch.sqrt(mse / N)
    # mae1 = mae1 / Npi
    # mse1 = torch.sqrt(mse1 / N)
    print(all)
    print(' * MAE {mae:.3f} \t    * MSE {mse:.3f}'
          .format(mae=mae, mse=mse))
    print(' * MAE1 {mae1:.3f} \t    * MSE1 {mse1:.3f}'
          .format(mae1=mae1, mse1=mse1))
main()