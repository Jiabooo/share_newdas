# encoding=UTF8
import scipy.io as scio
import os
import os.path as osp
import numpy as np
from PIL import Image
import sys
import glob
import pandas as pd
from swnms import *

def calculate_num(gtlist,length,width, gt):
    rowgt = len(gtlist)
    # print(rowgt)
    # print(gtlist.shape)
    distancex = length / 5
    distancey = width / 3
    pointgt = [0] * 15
    disgt = [[] for _ in range(15)]
    i = 0

    # pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    pts = gtlist

    # print(pts)
    print(pts.shape)

    # 构造KDTree寻找相邻的人头位置
    leafsize = 2048
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    distances, locations = tree.query(pts, k=4)

    while (i < rowgt):
        temgtpoint = gtlist[i]
        temgtx = temgtpoint[0]
        temgty = temgtpoint[1]

        # pt = pts[i]
        # pt2d = np.zeros(gt.shape, dtype=np.float32)
        # pt2d[pt[1],pt[0]] = 1.
        temdis = (distances[i][1] + distances[i][2] + distances[i][3])/3
        # print(temdis)

        i = i + 1
        if(temgtx >= length or temgty >= width):
            continue

        rowgtcount = temgtx // distancex
        colgtcount = temgty // distancey
        blockgtcount = colgtcount * 5 + rowgtcount
        blockgtcount = int(blockgtcount)

        pointgt[blockgtcount] = pointgt[blockgtcount] + 1
        # newdis = disgt[blockgtcount] * (pointgt[blockgtcount]-1) /(pointgt[blockgtcount])
        disgt[blockgtcount].append(temdis)

    return pointgt, disgt


root = r'D:\renqun\share_newdas\das\shanghai'

part_A_test =  os.path.join(root, 'part_A_final/test_data', 'images')

img_paths = []
for img_path in glob.glob(os.path.join(part_A_test,'*.jpg')):
    img_paths.append(img_path)

# for img_path in glob.glob(os.path.join(part_A_test,'IMG_109.jpg')):
#     img_paths.append(img_path)

sum = len(img_paths)
print('total len:' + str(sum))
total_point2dis = []

for i, img_path in enumerate(img_paths):
    print(img_path)
    mat_path = img_path.replace('images', 'ground_truth').replace('IMG','GT_IMG').replace('.jpg', '.mat')
    datagt = scio.loadmat(mat_path)
    gtlist = datagt['image_info'][0][0]
    gtlist = gtlist['location'][0][0]

    img = plt.imread(img_path)
    k = np.zeros((img.shape[0],img.shape[1]))
    gt = datagt["image_info"][0,0][0,0][0]
    for i in range(0,len(gt)):
        if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
            k[int(gt[i][1]),int(gt[i][0])]=1

    img = Image.open(img_path)
    tempointgt, temdisgt = calculate_num(gtlist, img.width, img.height, k)
    # temlist = list(zip(tempointgt,temdisgt))
    # print(zip(tempointgt,temdisgt))
    # print(list(zip(np.array(tempointgt),np.array(temdisgt))))
    # temdisgt = np.array(temdisgt)
    # print(temdisgt.shape)
    new_res = []
    for i in range(len(tempointgt)):
        # print(temdisgt[i])
        tem = [(tempointgt[i],j)for j in temdisgt[i]]
        tem = np.array(tem)
        # print(tem.shape)
        total_point2dis.append(tem)

    # print(tempointgt.shape)
    # print(temdisgt.shape)
    # total_point2dis.append()


# data_frame = pd.DataFrame({'data':total_point2dis[:]})
data_frame = pd.DataFrame(data=total_point2dis)
data_frame.to_csv("output_detailed.csv", index=False)
