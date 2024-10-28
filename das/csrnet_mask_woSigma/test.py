import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from skimage.feature import peak_local_max as plm
from scipy.ndimage import gaussian_filter
import scipy
import json
import torchvision.transforms.functional as F
from matplotlib import cm as CM
from image import *
from model import CSRNet
import torch
from matplotlib import cm as c
from torchvision import datasets, transforms
transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])
model = CSRNet()
#defining the model
model = model.cuda()
#loading the trained weights

# checkpoint = torch.load("/mnt/nvme1n1p1/Chenhao/6ceng_mask_63.0_102.5.tar")
checkpoint = torch.load("G:/renqun/das/das/csrnet_mask/new_mask.tar")
#checkpoint = torch.load(r"/mnt/nvme1n1p1/Chenhao/csrnet_mask_1e7/0\model_best.pth.tar")
model.load_state_dict(checkpoint['state_dict'])


num = 180
img = "G:/renqun/das/das/shanghai/part_A_final/test_data/images/IMG_{}.jpg".format(num)
#img = "/home/ch/SH_A/test_data/images/IMG_1.jpg"

temp = h5py.File(img.replace('.jpg','.h5').replace('images','ground_truth'), 'r')

target = np.asarray(temp['density'])
print("Original Count : ",int(np.sum(target)) + 1)
target = cv2.resize(target,(int(target.shape[1]/8),int(target.shape[0]/8)),interpolation = cv2.INTER_AREA)*64

plt.axis('off');
plt.imshow(target, cmap = c.jet)
plt.savefig("gt_{}.jpg".format(num), bbox_inches='tight',pad_inches = 0, dpi=300)

img1 = Image.open(img).convert('RGB')
img = transform(img1).cuda()
print(img.shape)
output,mask = model(img.unsqueeze(0))
num = int(output.detach().cpu().sum().numpy())
local_max = plm(img, min_distance=40, num_peaks=num, exclude_border=False)
for loc in local_max:
    for x in range(loc[0] - 9, loc[0] + 8):
        for y in range(loc[1] - 9, loc[1] + 8):
            img[x, y] = 255

signedPath = '预测点标注' + '.bmp'
cv2.imshow('预测点标注', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Predicted Count : ", int(output.detach().cpu().sum().numpy()))

#img = "/home/ch/csrnet_yanmo/1.jpg"
#img1 = plt.imread(img)


output = np.asarray(output.detach().cpu().reshape(output.detach().cpu().shape[2],output.detach().cpu().shape[3]))
plt.axis('off');
plt.imshow(output, cmap = c.jet)
plt.savefig("output_{}.jpg".format(num), bbox_inches='tight',pad_inches = 0, dpi=300)



mask = np.asarray(mask.detach().cpu().reshape(mask.detach().cpu().shape[2],mask.detach().cpu().shape[3]))
output = cv2.resize(output,(img.shape[2],img.shape[1]),interpolation = cv2.INTER_LINEAR)
mask = cv2.resize(mask,(img.shape[2],img.shape[1]),interpolation = cv2.INTER_LINEAR)

output = np.where(output>=0.1,1,0)


mask = np.where(mask>=0.1,1,0)
#mask = cv2.blur(mask, (15,15))
#mask = np.where(mask>=0.001,1,0)

#plt.axis('off');
#plt.imshow(mask, cmap = c.jet)
#plt.savefig("mask.jpg")

img1 = np.asarray(img1)
l = img1.copy()
for i in range(img1.shape[2]):
    l[:,:,i] = img1[:,:,i]*mask 
    
#plt.imshow(l)
#plt.savefig('1.jpg',dpi=200)

l = transform(l)
#print(l.shape)
output,mask = model(l.unsqueeze(0).cuda())

#print("Predicted Count : ", int(output.detach().cpu().sum().numpy()))
