import math
import shutil

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
from matplotlib import cm as CM
from skimage.feature import peak_local_max as plm
import cv2


def get_distance_array(ori_x, ori_y, img, new_img, new_x, new_y, input_low_limit, down):
    # 滑动窗口定位
    # 弃用
    windows_num_x = 3
    windows_num_y = 5

    step_x = int(ori_x/windows_num_x)
    step_y = int(ori_y/windows_num_y)
    # new_img = cv2.resize(np.float32(img), (new_y, new_x),interpolation=cv2.INTER_CUBIC)

    plt.axis('off')
    plt.imshow(new_img)
    output_distance_array = np.ones_like(new_img)

    # plt.show()
    output_dir = "res/"
    plt.savefig(output_dir + "test.jpg",dpi=300, bbox_inches='tight', pad_inches=0)
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

            # print(np.max(new_img[new_start_x:new_end_x, new_start_y:new_end_y]))
            # counting
            num = int(small_img.sum()/down)

            print("num" +str(num))

            if num == 0:
                continue

            # low_limit = 1
            # low_limit = 10
            low_limit = 8
            # low_limit = 1
            high_limit = 20

            # dis = int((step_x*step_y/num)/2)
            # dis = int(math.sqrt(new_step_x/8*new_step_y/8/num)/4)
            # dis = int(math.sqrt(new_step_x/8*new_step_y/8/num)/1)
            dis = int(math.sqrt(new_step_x/8*new_step_y/8/num)*2)
            # dis = int(math.sqrt(new_step_x/8*new_step_y/8/num)/2)
            # dis = int(math.sqrt(new_step_x/8*new_step_y/8/(num/2)))
            # dis = int(math.sqrt(new_step_x/8*new_step_y/8/num)/8)
            # dis = int(math.sqrt(new_step_x/8*new_step_y/8/num)/20)
            # dis = int(math.sqrt(new_step_x/8*new_step_y/8/num)/60)
            # dis = 1
            # dis = int(new_step_x/8*new_step_y/8/num/2)
            if dis < low_limit:
                dis = low_limit
            if dis > high_limit:
                dis = high_limit

            print(dis)

            output_distance_array[new_start_x:new_end_x,new_start_y:new_end_y] = dis

    return output_distance_array

def get_candidate(ori_x, ori_y, img, new_img, new_x, new_y, input_low_limit, down):
    # 滑动窗口定位
    windows_num_x = 3
    windows_num_y = 5

    step_x = int(ori_x/windows_num_x)
    step_y = int(ori_y/windows_num_y)
    # new_img = cv2.resize(np.float32(img), (new_y, new_x),interpolation=cv2.INTER_CUBIC)

    # plt.axis('off')
    # plt.imshow(new_img)
    output_distance_array = np.ones_like(new_img)

    # plt.show()
    # output_dir = "threshold_res/"
    # plt.savefig(output_dir + "test.jpg",dpi=300, bbox_inches='tight', pad_inches=0)
    # print("new_img:")
    # print(new_img.shape)
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

            # print(np.max(new_img[new_start_x:new_end_x, new_start_y:new_end_y]))
            if np.max(new_img[new_start_x:new_end_x, new_start_y:new_end_y]) < 0.05:
                continue

            # counting
            num = int(small_img.sum()/down)

            # print("num" +str(num))

            if num == 0:
                continue

            # low_limit = 1
            # low_limit = 10
            low_limit = 7
            # low_limit = 1
            high_limit = 20

            # dis = int((step_x*step_y/num)/2)
            # dis = int(math.sqrt(new_step_x/8*new_step_y/8/num)/4)
            # dis = int(math.sqrt(new_step_x/8*new_step_y/8/num)/1)
            dis = int(math.sqrt(new_step_x/8*new_step_y/8/num)*2)
            # dis = int(math.sqrt(new_step_x/8*new_step_y/8/num)/2)
            # dis = int(math.sqrt(new_step_x/8*new_step_y/8/(num/2)))
            # dis = int(math.sqrt(new_step_x/8*new_step_y/8/num)/8)
            # dis = int(math.sqrt(new_step_x/8*new_step_y/8/num)/20)
            # dis = int(math.sqrt(new_step_x/8*new_step_y/8/num)/60)
            # dis = 1
            # dis = int(new_step_x/8*new_step_y/8/num/2)
            if dis < low_limit:
                dis = low_limit
            if dis > high_limit:
                dis = high_limit

            # print(dis)

            output_distance_array[new_start_x:new_end_x,new_start_y:new_end_y] = dis

            # prefetch points
            local_max = plm(new_small_img, min_distance=dis, num_peaks=num, exclude_border=False)
            if num < 5:
                local_max = plm(new_small_img, min_distance=dis, num_peaks=num+2, exclude_border=False)
                # local_max = plm(new_small_img, min_distance=4, num_peaks=num+2, exclude_border=False)
            for loc in local_max:
                # print(loc)
                # print("value" + str(new_small_img[loc[0], loc[1]]))
                if new_small_img[loc[0], loc[1]] < 0.06:
                    continue
                # for x in range(loc[0] - 9, loc[0] + 8):
                #     for y in range(loc[1] - 9, loc[1] + 8):
                for x in range(loc[0], loc[0] + 1):
                    if x > new_x - 1:
                        continue
                    for y in range(loc[1], loc[1] + 1):
                        if y > new_y - 1:
                            continue
                        # img_position[0, x, y] = 255
                        # img_position[1, x, y] = 255
                        # img_position[2, x, y] = 255
                        new_img[x + new_start_x, y + new_start_y] += 10


    return output_distance_array,new_img


def have_step(ori_x, ori_y, img, new_img, new_x, new_y, input_low_limit, down):
    # 滑动窗口定位
    windows_num_x = 3
    windows_num_y = 5

    step_x = int(ori_x/windows_num_x)
    step_y = int(ori_y/windows_num_y)
    # new_img = cv2.resize(np.float32(img), (new_y, new_x),interpolation=cv2.INTER_CUBIC)

    plt.axis('off')
    plt.imshow(new_img)
    output_distance_array = np.ones_like(new_img)

    # plt.show()
    # output_dir = "res/"
    # plt.savefig(output_dir + "test.jpg",dpi=300, bbox_inches='tight', pad_inches=0)
    # print("new_img:")
    # print(new_img.shape)
    new_step_x = int(new_x/windows_num_x)
    new_step_y = int(new_y/windows_num_y)

    for i in range(windows_num_x-1):
        for j in range(windows_num_y-1):
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

            x_shift = int(step_x/2)
            y_shift = int(step_y/2)
            small_img = img[step_x * i + x_shift:end_x + x_shift,step_y * j + y_shift:end_y + y_shift]

            new_x_shift = int(new_step_x / 2)
            new_y_shift = int(new_step_y / 2)
            new_start_x = new_step_x * i
            new_start_y = new_step_y * j
            new_small_img = new_img[new_start_x+new_x_shift:new_end_x+new_x_shift,new_start_y+new_y_shift:new_end_y+new_y_shift]
            # print("new_small:i-"+ str(i) +"j-" + str(j))
            # print(new_small_img.shape)

            # print(np.max(new_small_img))
            if np.max(new_small_img) < 0.06:
                continue

            # counting
            num = int(small_img.sum()/down)

            # print("num" +str(num))

            if num == 0:
                continue

            # low_limit = 1
            # low_limit = 10
            # low_limit = 8
            low_limit = 7
            # low_limit = 1
            high_limit = 20

            # dis = int((step_x*step_y/num)/2)
            # dis = int(math.sqrt(new_step_x/8*new_step_y/8/num)/4)
            # dis = int(math.sqrt(new_step_x/8*new_step_y/8/num)/1)
            dis = int(math.sqrt(new_step_x/8*new_step_y/8/num)*2)
            # dis = int(math.sqrt(new_step_x/8*new_step_y/8/num)/2)
            # dis = int(math.sqrt(new_step_x/8*new_step_y/8/(num/2)))
            # dis = int(math.sqrt(new_step_x/8*new_step_y/8/num)/8)
            # dis = int(math.sqrt(new_step_x/8*new_step_y/8/num)/20)
            # dis = int(math.sqrt(new_step_x/8*new_step_y/8/num)/60)
            # dis = 1
            # dis = int(new_step_x/8*new_step_y/8/num/2)
            if dis < low_limit:
                dis = low_limit
            if dis > high_limit:
                dis = high_limit

            # print(dis)

            # output_distance_array[new_start_x+new_x_shift:new_end_x+new_x_shift,new_start_y+new_y_shift:new_end_y+new_y_shift] = dis

            # prefetch points
            local_max = plm(new_small_img, min_distance=dis, num_peaks=num, exclude_border=False)
            if num < 5:
                local_max = plm(new_small_img, min_distance=dis, num_peaks=num+2, exclude_border=False)
            for loc in local_max:
                # print("value" + str(new_small_img[loc[0], loc[1]]))
                if new_small_img[loc[0], loc[1]] < 0.06:
                    continue
                # for x in range(loc[0] - 9, loc[0] + 8):
                #     for y in range(loc[1] - 9, loc[1] + 8):
                for x in range(loc[0], loc[0] + 1):
                    if x > new_x - 1:
                        continue
                    for y in range(loc[1], loc[1] + 1):
                        if y > new_y - 1:
                            continue
                        # img_position[0, x, y] = 255
                        # img_position[1, x, y] = 255
                        # img_position[2, x, y] = 255
                        new_img[x + new_start_x + new_x_shift, y + new_start_y+ new_y_shift] += 10


    return new_img