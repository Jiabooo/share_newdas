from PIL import Image
import numpy as np
import cv2
import os
import glob
from torchvision import transforms
import torch
import json
# 修改json文件
files = ['A_train', 'A_test', 'B_train', 'B_test']
for file in files:
    json_p = 'D:/renqun/new_das/new_das/das/csrnet_mask/' + str(file) + '.json'
    with open(json_p, 'r', encoding='utf-8') as f:
        s = f.read()

    s = s.replace('D:/renqun/new_das/new_das/das', "D:/renqun/new_das/new_das/das/")  ##字符串正则转换
    json_out = json.loads(s)

    json_np = json_p
    with open(json_np, 'w', encoding='utf-8') as f:
        json.dump(json_out, f)
