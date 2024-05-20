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
from model1 import CSRNet1
import torch
from matplotlib import cm as c
from torchvision import datasets, transforms
import cv2
from torch.autograd import Variable

def locate_people(ori_x, ori_y, img, new_x, new_y, input_low_limit):
    # 滑动窗口定位
    windows_num_x = 3
    windows_num_y = 5
    # windows_num_x = 5
    # windows_num_y = 8
    # windows_num_x = 8
    # windows_num_y = 12
    step_x = int(ori_x/windows_num_x)
    step_y = int(ori_y/windows_num_y)
    new_img = cv2.resize(np.float32(img), (new_y, new_x),interpolation=cv2.INTER_AREA)
    print("new_img:")
    print(new_img.shape)
    new_step_x = int(new_x/windows_num_x)
    new_step_y = int(new_y/windows_num_y)
    for i in range(windows_num_x):
        for j in range(windows_num_y):
            # create window
            if i == windows_num_x -1:
                end_x = ori_x
                new_end_x = new_x
            else:
                end_x = step_x*(i+1)
                new_end_x = new_step_x*(i+1)
            if j == windows_num_y -1:
                end_y = ori_y
                new_end_y = new_y
            else:
                end_y = step_y * (j + 1)
                new_end_y = new_step_y * (j + 1)
            small_img = img[step_x * i:end_x,step_y * j:end_y]

            new_start_x = new_step_x * i
            new_start_y = new_step_y * j
            new_small_img = new_img[new_start_x:new_end_x,new_start_y:new_end_y]
            # print("new_small:i-"+ str(i) +"j-" + str(j))
            # print(new_small_img.shape)

            # counting
            num = int(small_img.sum())

            if num == 0:
                continue

            low_limit = 2
            high_limit = 20

            # dis = int((step_x*step_y/num)/2)
            dis = int(new_step_x/8*new_step_y/8/num)
            if dis < low_limit:
                dis = low_limit
            if dis > high_limit:
                dis = high_limit

            # prefetch points
            local_max = plm(new_small_img, min_distance=dis, num_peaks=num, exclude_border=False)
            for loc in local_max:
                # for x in range(loc[0] - 9, loc[0] + 8):
                #     for y in range(loc[1] - 9, loc[1] + 8):
                for x in range(loc[0], loc[0] + 1):
                    if x > new_x-1:
                        continue
                    for y in range(loc[1], loc[1] + 1):
                        if y > new_y-1:
                            continue
                        # img_position[0, x, y] = 255
                        # img_position[1, x, y] = 255
                        # img_position[2, x, y] = 255
                        new_img[x+new_start_x, y+new_start_y] = 100000

    total_num = int(img.sum())

    print(total_num)
    low_limit = input_low_limit
    high_limit = 20
    dis = int(((new_x/8 * new_y/8) / total_num) / 4)
    # dis = int(((new_x/8 * new_y/8)/ total_num))
    # dis = int(((step_x * step_y) / total_num**2) / 2)
    #dis = int(((new_x/8 + new_y/8) / total_num**2) / 8)
    if dis < low_limit:
        dis = low_limit
    if dis > high_limit:
        dis = high_limit
    # final locate
    local_max = plm(new_img, min_distance=dis, num_peaks=total_num, exclude_border=False)
    img_position = np.zeros((new_x, new_y, 3))
    # img_position = img.cpu.numpy()

    # img_position = output
    for loc in local_max:
        # for x in range(loc[0] - 9, loc[0] + 8):
        #     for y in range(loc[1] - 9, loc[1] + 8):
        for x in range(loc[0]-3, loc[0] + 4):
            if x > new_x-1:
                continue
            for y in range(loc[1]-3, loc[1] + 4):
                if y > new_y-1:
                    continue
                img_position[x, y, 0] = 255
                img_position[x, y, 1] = 255
                img_position[x, y, 2] = 255

    return img_position


transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])
# model = CSRNet()
# #defining the model
# model = model.cuda()
# #loading the trained weights


model = CSRNet()
# pretrained = torch.load(r"D:\renqun\share_newdas\das\csrnet_mask\new_mask.tar")
# pretrained = torch.load(r"D:\renqun\share_newdas\das\mask_depth2\mask_depth.tar")
pretrained = torch.load(r"D:\renqun\share_newdas\das\mask_depth2\mask_depth.tar")
# pretrained = torch.load(r"D:\renqun\share_newdas\das\mask_depth2\0model_best.pth.tar")
model = model.cuda()
model.load_state_dict(pretrained['state_dict'])

mask_model = CSRNet1()
# pretrained = torch.load(r"D:\renqun\share_newdas\das\csrnet_mask\new_mask.tar")
# pretrained = torch.load(r"D:\renqun\share_newdas\das\mask_depth2\mask_depth.tar")
pretrained = torch.load(r"D:\renqun\share_newdas\das\mask_depth2\mask_depth.tar")
# pretrained = torch.load(r"D:\renqun\share_newdas\das\mask_depth2\0model_best.pth.tar")
mask_model = mask_model.cuda()
mask_model.load_state_dict(pretrained['state_dict'])



#checkpoint = torch.load(r"/public/home/aceukas6qi/das/mask_depth/mask_depth.tar")
#model.load_state_dict(checkpoint['state_dict'])


pic_num = 30
img_path = r"D:\renqun\share_newdas\das\shanghai\part_A_final/test_data/images/IMG_{}.jpg".format(pic_num)
#img = "/home/ch/SH_A/test_data/images/IMG_1.jpg"

temp = h5py.File(img_path.replace('.jpg','.h5').replace('images','ground_truth'), 'r')

temp_1 = np.asarray(temp['density'])
print("Original Count : ",int(np.sum(temp_1)) + 1)


img1 = Image.open(img_path).convert('RGB')
img = transform(img1).cuda()
print(img.shape)
print(img.unsqueeze(0))

# output,mask = model(img.unsqueeze(0))
# num = int(output.detach().cpu().sum().numpy())

# get depth
gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')
gt_file = h5py.File(gt_path)
depth_target = np.asarray(gt_file['density'])
depth_target = np.clip(depth_target, 0, 50)
depth_target = np.min(depth_target) + np.max(depth_target) - depth_target
depth_target = depth_target  # -depth_target
depth_target = cv2.resize(np.float32(depth_target),(int(depth_target.shape[1]/8),int(depth_target.shape[0]/8)),interpolation = cv2.INTER_AREA)

depth = depth_target

model.eval()
mask_model.eval()
num = 0     #预计人数

img = img.cuda()
img = Variable(img)

output1, mask1 = mask_model(img)
# output1 = output1/10
mask1 = torch.where(mask1 > 0.01, 1, 0)
output1 = torch.where(output1 > 0.01, 1, 0)
depth = torch.Tensor(depth).type(torch.FloatTensor).unsqueeze(0).cuda() * output1

# output, mask = model(img, mask1, depth)
output, mask = model(img, depth,mask1)
# num = int((output.data.sum()/10).cpu().numpy())
num = int((output.data.sum()).cpu().numpy())


output = np.asarray(output.detach().cpu().reshape(output.detach().cpu().shape[1],output.detach().cpu().shape[2]))
print(output)

new_img = plt.imread(img_path)
plt.imshow(new_img)
# plt.show()

img_test = Image.open(img_path).convert('RGB')
img_test2 = transform(img_test)

# test_1 = output
# test_1.fill(1)
# loc_output= test_1 - output     #取反


img_position = np.asarray(img_test)
print(img_position.shape)
img_position = locate_people(output.shape[0],  output.shape[1], output, img_position.shape[0], img_position.shape[1], 2)

# print(img_position)
print("Predicted Count : ", num)

plt.axis('off')
plt.imshow(img_position)

# plt.show()
plt.savefig("my_output_position{}.jpg".format(pic_num),dpi=300,bbox_inches='tight', pad_inches=0)



# gt
img_position = locate_people(img_position.shape[0],  img_position.shape[0], temp_1, img_position.shape[0], img_position.shape[1], 1)

plt.axis('off')
plt.imshow(img_position)

# plt.show()
plt.savefig("my_gt_output_position{}.jpg".format(pic_num),dpi=300,bbox_inches='tight', pad_inches=0)

local_max = plm(temp_1, min_distance=2, num_peaks=num, exclude_border=False)
# print(local_max)
# img_position = img.cpu.numpy()
img_position = np.array(img_test)
# img_position = output
img_position.fill(0)
# print(img_position)
# print(img_position.shape)
# print(output.shape)
for loc in local_max:
    # for x in range(loc[0] - 9, loc[0] + 8):
    #     for y in range(loc[1] - 9, loc[1] + 8):
    for x in range(loc[0]-3, loc[0]+4):
        if x > img_position.shape[0]-1:
            continue
        for y in range(loc[1]-3, loc[1]+4):
            if y > img_position.shape[1]-1:
                continue
            # img_position[0, x, y] = 255
            # img_position[1, x, y] = 255
            # img_position[2, x, y] = 255
            img_position[x, y, 0] = 255
            img_position[x, y, 1] = 255
            img_position[x, y, 2] = 255
            # img_position[x, y] = 1.0

print(img_position)
# img_position = img_position.permute(1,2,0) #交换维度

# signedPath = '预测点标注' + '.bmp'
# cv2.imshow('预测点标注', img_position)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
print("Predicted Count : ", num)

# print(img_position.detach().cpu().shape)
# img_position = np.asarray(img_position.detach().cpu().reshape(img_position.detach().cpu().shape[1],img_position.detach().cpu().shape[2]))

plt.axis('off')
plt.imshow(img_position)

# plt.show()
plt.savefig("gt_output_position{}.jpg".format(pic_num),dpi=300,bbox_inches='tight', pad_inches=0)
