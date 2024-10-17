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
from skimage.feature import peak_local_max as plm
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


#checkpoint = torch.load(r"/public/home/aceukas6qi/das/mask_depth/mask_depth.tar")
#model.load_state_dict(checkpoint['state_dict'])


num = 119
img = r"G:\renqun\das\das\shanghai\part_B_final/test_data/images/IMG_{}.jpg".format(num)
#img = "/home/ch/SH_A/test_data/images/IMG_1.jpg"

temp = h5py.File(img.replace('.jpg','.h5').replace('images','ground_truth'), 'r')

temp_1 = np.asarray(temp['density'])
print("Original Count : ",int(np.sum(temp_1)) + 1)


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
plt.savefig("output{}.jpg".format(num),dpi=300,bbox_inches='tight', pad_inches=0)


