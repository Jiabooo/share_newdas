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
input_dir = "predicted_density/A/"
output_dir = "threshold_res_swnmsOnly/"
if not os.path.exists(output_dir):  # 如果路径不存在
    os.makedirs(output_dir)
# output_npy_dir = "threshold_res_depth_B_npy/"
output_npy_dir = "threshold_res_swnmsOnly_npy/"
if not os.path.exists(output_npy_dir):  # 如果路径不存在
    os.makedirs(output_npy_dir)
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

    output = np.load(input_dir + "Density_{}.npy".format(pic_num))
    num = int(np.sum(output))
    print("num" + str(num))

    img = Image.open(img_path).convert('RGB')
    img_position = np.asarray(img)


    # test_1 = output
    # test_1.fill(1)
    # loc_output= test_1 - output     #取反

    if allow_print:
        print("output" + str(output.shape))

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
    # print(new_img.shape)

    test_distance, new_candidates = get_candidate(output.shape[0], output.shape[1], output, new_img,
                                                  img_position.shape[0], img_position.shape[1], 1, 1)
    new_candidates = have_step(output.shape[0], output.shape[1], output, new_candidates, img_position.shape[0],
                               img_position.shape[1], 1, 1)


    coordinates = peak_local_max(new_candidates, distance_array=test_distance, num_peaks=num, exclude_border=False)

    # print("coordinates" + str(coordinates.shape))
    assert coordinates.shape[0] == num
    out_coordinates = coordinates[:,[1, 0]]
    # out_coordinates[:, 1] = img_position.shape[1] - out_coordinates[:, 1]
    np.save(output_npy_dir + "IMG_{}".format(pic_num), out_coordinates, allow_pickle=True, fix_imports=True)


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

# for pic_num in range(27, 182+1):
#     print(pic_num)
#     run_test_location(pic_num)

for pic_num in range(78, 182+1):
    if(pic_num % 10 == 0):
        # torch.cuda.empty_cache()
        gc.collect(generation=2)
    print(pic_num)
    run_test_location(pic_num)

# for pic_num in range(88, 316+1):
#     # if(pic_num % 10 == 0):
#     #     torch.cuda.empty_cache()
#     #     gc.collect()
#     print(pic_num)
#     run_test_location(pic_num)

# run_test_location(30)
# run_test_location(54)