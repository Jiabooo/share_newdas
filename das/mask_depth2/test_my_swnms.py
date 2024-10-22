from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from swnms_utils_peak import peak_local_max
from skimage import data, img_as_float
import numpy as np
from swnms import *
import math
import shutil

import h5py
import scipy.io as io
import PIL.Image as Image
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
import gc

transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])

# output_dir = "threshold_res_depth_B/"
output_dir = "threshold_res_depth-down/"
# output_npy_dir = "threshold_res_depth_B_npy/"
output_npy_dir = "threshold_res_depth-down_npy/"
output_model = "A"
allow_print = False


model = CSRNet()
# pretrained = torch.load(r"D:\renqun\share_newdas\das\csrnet_mask\new_mask.tar")
# pretrained = torch.load(r"D:\renqun\share_newdas\das\mask_depth2\mask_depth.tar")
# pretrained = torch.load(r"D:\renqun\share_newdas\das\mask_depth2\mask_depth.tar")
pretrained = torch.load(r"D:\renqun\share_newdas\das\mask_depth2\ressultModels\A_2model_best.pth.tar")
# pretrained = torch.load(r"D:\renqun\share_newdas\das\mask_depth2\ressultModels\Bmodel_best.pth.tar")
# pretrained = torch.load(r"D:\renqun\share_newdas\das\mask_depth2\result_our_newmask.tar")
# pretrained = torch.load(r"D:\renqun\share_newdas\das\mask_depth2\0model_best.pth.tar")
model = model.cuda()
model.load_state_dict(pretrained['state_dict'])

mask_model = CSRNet1()
# pretrained = torch.load(r"D:\renqun\share_newdas\das\csrnet_mask\new_mask.tar")
# pretrained = torch.load(r"D:\renqun\share_newdas\das\mask_depth2\mask_depth.tar")
# pretrained = torch.load(r"D:\renqun\share_newdas\das\mask_depth2\mask_depth.tar")
pretrained = torch.load(r"D:\renqun\share_newdas\das\mask_depth2\ressultModels\second_Amodel_best.pth.tar")
# pretrained = torch.load(r"D:\renqun\share_newdas\das\mask_depth2\ressultModels\second_Bmodel_best.pth.tar")
# pretrained = torch.load(r"D:\renqun\share_newdas\das\mask_depth2\result_our_newmask.tar")
# pretrained = torch.load(r"D:\renqun\share_newdas\das\mask_depth2\0model_best.pth.tar")
mask_model = mask_model.cuda()
mask_model.load_state_dict(pretrained['state_dict'])



#checkpoint = torch.load(r"/public/home/aceukas6qi/das/mask_depth/mask_depth.tar")
#model.load_state_dict(checkpoint['state_dict'])




def run_test_location(pic_num):
    img_path = r"D:\renqun\share_newdas\das\shanghai\part_A_final/test_data/images/IMG_{}.jpg".format(pic_num)
    # img_path = r"D:\renqun\share_newdas\das\shanghai\part_B_final/test_data/images/IMG_{}.jpg".format(pic_num)
    # img = "/home/ch/SH_A/test_data/images/IMG_1.jpg"

    # temp = h5py.File(img_path.replace('.jpg','.h5').replace('images','ground_truth'), 'r')
    #
    # temp_1 = np.asarray(temp['density'])
    # print("Original Count : ",int(np.sum(temp_1)) + 1)

    img1 = Image.open(img_path).convert('RGB')
    img = transform(img1).cuda()
    if allow_print:
        print(img.shape)
    # print(img.unsqueeze(0))

    # output,mask = model(img.unsqueeze(0))
    # num = int(output.detach().cpu().sum().numpy())

    # get depth
    # gt_path = img_path.replace('.jpg', '.h5').replace('images', 'depth_density_map')
    # gt_file = h5py.File(gt_path)
    # depth_target = np.asarray(gt_file['density'])
    # depth_target = np.clip(depth_target, 0, 50)
    # depth_target = np.min(depth_target) + np.max(depth_target) - depth_target
    # depth_target = depth_target  # -depth_target
    # depth_target = cv2.resize(np.float32(depth_target),(int(depth_target.shape[1]/8),int(depth_target.shape[0]/8)),interpolation = cv2.INTER_AREA)

    # depth = depth_target

    down = 10
    # my
    allow_depth = True
    allow_density_as_depth = True

    if allow_depth:
        depth_path = img_path.replace('.jpg', '.h5').replace('images', 'depth_density_map')
        depth = h5py.File(depth_path)
        depth = np.asarray(depth['density'])
        # depth = depth*down
        depth = cv2.resize(np.float32(depth),
                           (int(depth.shape[1] / 8), int(depth.shape[0] / 8)),
                           interpolation=cv2.INTER_AREA)


        depth = np.clip(depth, 0, 50)
        depth = np.min(depth) + np.max(depth) - depth

        # print(depth)
        # plt.axis('off')
        # plt.imshow(depth)
        # plt.show()
        # depth = cv2.resize(np.float32(depth),
        #                    (int(depth.shape[1] / 8), int(depth.shape[0] / 8)),
        #                    interpolation=cv2.INTER_AREA)

        model.eval()
        mask_model.eval()
        num = 0  # 预计人数

        img = img.cuda()
        img = Variable(img)


        output1, mask1 = mask_model(img)
        output1 = output1 / down
        mask1 = torch.where(mask1 > 0.01, 1, 0)
        output1 = torch.where(output1 > 0.01, 1, 0)
        depth = torch.Tensor(depth).type(torch.FloatTensor).cuda() * output1
        # print(depth)
        # plt.axis('off')
        # plt.imshow(depth.cpu())
        # plt.show()
        output, mask = model(img, depth, mask1)


        # with torch.no_grad():
        #     output1, mask1 = mask_model(img)
        #     # output1 = output1 / down
        #     mask1 = torch.where(mask1 > 0.01, 1, 0)
        #     output1 = torch.where(output1 > 0.01, 1, 0)
        #     depth = torch.Tensor(depth).type(torch.FloatTensor).unsqueeze(0).cuda() * output1
        #
        #     output, mask = model(img, mask1, depth)

        if allow_print:
            print(depth.shape)


    elif allow_density_as_depth:
        model.eval()
        mask_model.eval()
        num = 0  # 预计人数

        img = img.cuda()
        img = Variable(img)

        output1, mask1 = mask_model(img)
        # output, mask1 = mask_model(img)

        # print(output1)
        new_output = np.asarray(
            output1.detach().cpu().reshape(output1.detach().cpu().shape[1], output1.detach().cpu().shape[2]))
        # print(new_output)
        depth_target = np.clip(new_output, 0, 50)
        depth_target = np.min(depth_target) + np.max(depth_target) - depth_target
        # print(depth_target)
        plt.axis('off')
        plt.imshow(depth_target)
        plt.show()

        mask1 = torch.where(mask1 > 0.01, 1, 0)
        new_output1 = torch.where(output1 > 0.01, 1, 0)
        if allow_print:
            print(output1.shape)

        # pre_depth = torch.Tensor(depth).type(torch.FloatTensor).cuda() * output1
        depth = torch.Tensor(depth_target).type(torch.FloatTensor).cuda() * new_output1
        # print(depth)

        # output, mask = model(img, mask1, depth)

        output, mask = model(img, depth, mask1)
        # num = int((output.data.sum()/10).cpu().numpy())

    else:
        mask_model.eval()
        output, mask1 = mask_model(img)

    if allow_print:
        print(output)
    num = int((output.data.sum()).cpu().numpy())
    # num = int((output.data.sum()).cpu().numpy()/10)

    output = np.asarray(output.detach().cpu().reshape(output.detach().cpu().shape[1], output.detach().cpu().shape[2]))
    # print(output)

    new_img = plt.imread(img_path)
    if allow_print:
        print("new_img" + str(new_img.shape))
    # plt.imshow(new_img)
    # plt.show()

    img_test = Image.open(img_path).convert('RGB')
    img_test2 = transform(img_test)

    # test_1 = output
    # test_1.fill(1)
    # loc_output= test_1 - output     #取反

    if allow_print:
        print("output" + str(output.shape))
    img_position = np.asarray(img_test)
    if allow_print:
        print("img_position" + str(img_position.shape))
    # print("img_position" + str(img_position.shape))
    # print("img_position" + str(img_position.shape))
    # new_img = cv2.resize(np.float32(output), (img_position.shape[1], img_position.shape[0]),interpolation=cv2.INTER_AREA)
    # new_img = cv2.resize(np.float32(output), (img_position.shape[1], img_position.shape[0]),
    #                      interpolation=cv2.INTER_NEAREST)
    new_img = cv2.resize(np.float32(output), (img_position.shape[1], img_position.shape[0]),
                         interpolation=cv2.INTER_AREA)
    # new_img = output
    if allow_print:
        print("new_img" + str(new_img.shape))
    # img_position = locate_people(output.shape[0],  output.shape[1], output, img_position.shape[0], img_position.shape[1], 3, 10)
    # img_position = locate_people(output.shape[0],  output.shape[1], output, img_position.shape[0], img_position.shape[1], 3, 1)
    # test_distance = get_distance_array(output.shape[0],  output.shape[1], output, new_img, img_position.shape[0], img_position.shape[1], 1, 1)

    # print the output density
    plt.axis('off')
    plt.imshow(new_img)
    plt.savefig(output_dir + "{}_{}_my_output_density{}_{}.jpg".format(pic_num, output_model, pic_num, num), dpi=300,
                bbox_inches='tight', pad_inches=0)

    test_distance, new_candidates = get_candidate(output.shape[0], output.shape[1], output, new_img,
                                                  img_position.shape[0], img_position.shape[1], 1, 1)
    new_candidates = have_step(output.shape[0], output.shape[1], output, new_candidates, img_position.shape[0],
                               img_position.shape[1], 1, 1)

    # print(test_distance)
    # print(test_distance.shape)

    # im = img_as_float(data.coins())

    # img_position = get_distance_array(output.shape[0],  output.shape[1], output, img_position.shape[0], img_position.shape[1], 1, 1)

    # shape = im.shape

    # array_20 = np.full(shape, 20)

    # test_distance = get_distance_array(output.shape[0],  output.shape[1], output, img_position.shape[0], img_position.shape[1], 1, 1)

    # image_max is the dilation of im with a 20*20 structuring element
    # It is used within peak_local_max function
    # image_max = ndi.maximum_filter(im, size=20, mode='constant')

    # Comparison between image_max and im to find the coordinates of local maxima
    # coordinates = peak_local_max(im, distance_array=array_20)
    coordinates = peak_local_max(new_candidates, distance_array=test_distance, num_peaks=num, exclude_border=False)

    # print("coordinates" + str(coordinates.shape))
    out_coordinates = coordinates[:,[1, 0]]
    # out_coordinates[:, 1] = img_position.shape[1] - out_coordinates[:, 1]
    np.save(output_npy_dir + "IMG_{}".format(pic_num), out_coordinates, allow_pickle=True, fix_imports=True)

    # display results
    # fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
    # ax = axes.ravel()
    # ax[0].imshow(im, cmap=plt.cm.gray)
    # ax[0].axis('off')
    # ax[0].set_title('Original')
    #
    # ax[1].imshow(image_max, cmap=plt.cm.gray)
    # ax[1].axis('off')
    # ax[1].set_title('Maximum filter')
    #
    # ax[2].imshow(im, cmap=plt.cm.gray)
    # ax[2].autoscale(False)
    # ax[2].plot(coordinates[:, 1], coordinates[:, 0], 'r.')
    # ax[2].axis('off')
    # ax[2].set_title('Peak local max')

    # fig.tight_layout()
    #
    # plt.show()

    img_position = np.zeros_like(new_img)
    # print("img_position" + str(img_position.shape))

    # for loc in coordinates:
    #     for x in range(loc[0] - 3, loc[0] + 4):
    #         if x > img_position.shape[0] - 1 or x < 0:
    #             continue
    #         for y in range(loc[1] - 3, loc[1] + 4):
    #             if y > img_position.shape[1] - 1 or y < 0:
    #                 continue
    #             img_position[x, y, 0] = 255
    #             img_position[x, y, 1] = 255
    #             img_position[x, y, 2] = 255

    for loc in coordinates:
        for x in range(loc[0] - 3, loc[0] + 4):
            if x > img_position.shape[0] - 1 or x < 0:
                continue
            for y in range(loc[1] - 3, loc[1] + 4):
                if y > img_position.shape[1] - 1 or y < 0:
                    continue
                img_position[x, y] = 1

    # for loc in coordinates:
    #     for x in range(loc[0], loc[0] + 1):
    #         if x > img_position.shape[0] - 1 or x < 0:
    #             continue
    #         for y in range(loc[1], loc[1] + 1):
    #             if y > img_position.shape[1] - 1 or y < 0:
    #                 continue
    #             # img_position[x, y, 0] = 255
    #             # img_position[x, y, 1] = 255
    #             # img_position[x, y, 2] = 255
    #             img_position[x, y] = 1

    plt.axis('off')
    plt.imshow(img_position)

    # plt.show()
    plt.savefig(output_dir + "{}_{}_my_output_position{}_{}.jpg".format(pic_num, output_model, pic_num, num), dpi=300,
                bbox_inches='tight', pad_inches=0)

    # 原图输出
    shutil.copy(img_path, output_dir + "{}_{}_ori_{}.jpg".format(pic_num, output_model, pic_num))






# pic_num = 39

for pic_num in range(160, 182+1):
    if(pic_num % 10 == 0):
        torch.cuda.empty_cache()
    print(pic_num)
    run_test_location(pic_num)

# for pic_num in range(130, 182+1):
#     # if(pic_num % 10 == 0):
#     #     torch.cuda.empty_cache()
#     #     gc.collect()
#     print(pic_num)
#     run_test_location(pic_num)

# for pic_num in range(88, 316+1):
#     # if(pic_num % 10 == 0):
#     #     torch.cuda.empty_cache()
#     #     gc.collect()
#     print(pic_num)
#     run_test_location(pic_num)

# run_test_location(19)