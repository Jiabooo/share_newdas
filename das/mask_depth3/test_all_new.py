#encoding=UTF8
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

transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]),
])


def main():
    # with open("A_test.json", 'r') as outfile:
    #     test_list = json.load(outfile)
    with open("../mask_depth_woSigma/B_test.json", 'r') as outfile:
        test_list = json.load(outfile)

    model = CSRNet()
    # pretrained = torch.load(r"D:\renqun\share_newdas\das\mask_depth2\ressultModels\A_2model_best.pth.tar") # 最后那个best模型的路径
    pretrained = torch.load(r"D:\renqun\share_newdas\das\mask_depth2\ressultModels\Bmodel_best.pth.tar")
    model = model.cuda()
    model.load_state_dict(pretrained['state_dict'])

    mask_model = CSRNet1()
    # pretrained = torch.load(r"D:\renqun\share_newdas\das\mask_depth2\ressultModels\second_Amodel_best.pth.tar") # second_A路径，用于掩膜
    pretrained = torch.load(r"D:\renqun\share_newdas\das\mask_depth2\ressultModels\second_Bmodel_best.pth.tar")
    mask_model = mask_model.cuda()
    mask_model.load_state_dict(pretrained['state_dict'])

    output_density_dir = "predicted_density/B/"
    if not os.path.exists(output_density_dir):  # 如果路径不存在
        os.makedirs(output_density_dir)

    test(test_list, model, mask_model, output_density_dir)




def test(test_list, model, mask_model, output_density_dir):
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
    all = 0
    for i, (img_path, img,target, count_target, mask_target,depth) in enumerate(test_loader):
        img = img.cuda()
        img = Variable(img)

        with torch.no_grad():
            output1, mask1 = mask_model(img)
            # output1 = output1/down
            mask1 = torch.where(mask1 > 0.01, 1, 0)
            output1 = torch.where(output1 > 0.01, 1, 0)
            depth = depth.type(torch.FloatTensor).unsqueeze(0).cuda() * output1

            # output,mask = model(img,mask1,depth)
            output, mask = model(img, depth, mask1)  # normal forword
            # output = output / down

        # target_sum = (target.sum().type(torch.FloatTensor).cuda() + count_target.sum().type(
        # torch.FloatTensor).cuda()) / 2
        target_sum = target.sum().type(torch.FloatTensor).cuda()
        mae += abs(output.data.sum() - target_sum)
        mse += (output.data.sum() - target_sum).pow(2)

        # 保存图像
        # print(output.shape)
        # print(img_path)
        fileName = os.path.basename(img_path[0])
        pic_num = fileName[4:-4]
        print(pic_num)
        output = output.cpu().numpy()
        output = np.asarray(output.reshape(output.shape[2], output.shape[3]))
        # print(output)
        print("progress:" + str(i+1))
        np.save(output_density_dir + "Density_{}.npy".format(pic_num), output)

    N = len(test_loader)
    mae = mae / N
    mse = torch.sqrt(mse / N)
    print(all)
    print(' * MAE {mae:.3f} \t    * MSE {mse:.3f}'
          .format(mae=mae, mse=mse))


main()