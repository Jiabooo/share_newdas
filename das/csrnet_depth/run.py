"""Compute depth maps for images in the input folder.
"""
import os
import glob
import torch
import utils
import cv2
import argparse
from matplotlib import cm as CM
from torchvision.transforms import Compose
from midas.dpt_depth import DPTDepthModel
from midas.midas_net import MidasNet
from midas.midas_net_custom import MidasNet_small
from midas.transforms import Resize, NormalizeImage, PrepareForNet
from matplotlib import pyplot as plt
import numpy as np
import time
import scipy.io as io
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy import spatial
import time
from PIL import Image,ImageFilter,ImageDraw
import time
from pylab import *
from matplotlib import cm as CM
import h5py
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

import h5py

import scipy.io as io

from scipy.ndimage import gaussian_filter
from scipy import spatial



from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures as p



def gaussian_filter_density(gt,depth):   
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)  # nonzero value represent people in labels
    if gt_count == 0:  # gt_count is the amount of people
        return density
    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))  # human label position

    leafsize = 2048
    tree = spatial.KDTree(pts.copy(), leafsize=leafsize)
    distances, locations = tree.query(pts, k=4)

    print('generate density...')
    x=[]
    y=[]
    print(len(pts))
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
            if sigma > 5:
                sigma = 5
        else:
            sigma = np.average(np.array(gt.shape))/2./2.  
        
        x.append(depth[pt[1],pt[0]])
        y.append(sigma)
    
    x= np.array(x)
    #print(x)
    X = x.reshape(-1,1)

    #print(X)

    y= np.array(y)
    
    #1ci
    #model = linear_model.LinearRegression()
    #model.fit(X,y)
    #y_p = model.predict(X)
    
    #2ci
    qf = p(degree=2)
    z = qf.fit_transform(X)
    model = linear_model.LinearRegression()

    # myself
    try:
        model.fit(z, y)
    except:
        model.fit(z, y)

    #model.fit(z, y)
    y_p = model.predict(z)
    
    density = np.zeros(gt.shape, dtype=np.float32)
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        sigma = y_p[i]*0.5
        density += gaussian_filter(pt2d, sigma, mode='constant')        
    print('done.')
    return density
  
def Smooth_heaviside(x):
    x1 = 2 - 1 / (torch.sigmoid(1e7 * x) )
    x2 = torch.sigmoid(1e7 * x)
    return  x2*x1  
    
def run(input_path, output_path, model_path, model_type="large", optimize=True):
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
    """
    print("initialize")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    # load network
    if model_type == "dpt_large": # DPT-Large
        model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "dpt_hybrid": #DPT-Hybrid
        model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode="minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "midas_v21":
        model = MidasNet(model_path, non_negative=True)
        net_w, net_h = 384, 384
        resize_mode="upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    elif model_type == "midas_v21_small":
        model = MidasNet_small(model_path, features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={'expand': True})
        net_w, net_h = 256, 256
        resize_mode="upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        print(f"model_type '{model_type}' not implemented, use: --model_type large")
        assert False
    
    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    model.eval()
    
    if optimize==True:
        # rand_example = torch.rand(1, 3, net_h, net_w)
        # model(rand_example)
        # traced_script_module = torch.jit.trace(model, rand_example)
        # model = traced_script_module

        if device == torch.device("cuda"):
            model = model.to(memory_format=torch.channels_last)
            model = model.half()

    model.to(device)
    # get input
    img_names = glob.glob(os.path.join(input_path, "*"))
    num_images = len(img_names)
    print(num_images)
    #img_names=[]
    #img_names.append(input_path)
    #num_images = len(img_names)
    
    # create output folder
    #os.makedirs(output_path, exist_ok=True)

    print("start processing")

    for ind, img_name in enumerate(img_names):

        print("  processing {} ({}/{})".format(img_name, ind, num_images))

        # input

        img = utils.read_image(img_name)
        img_input = transform({"image": img})["image"]

        # compute
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
            if optimize==True and device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)
                sample = sample.half()
            prediction = model.forward(sample)
            depth = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        print(torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ))
        print(depth)

        #gt_path = img_name.replace('.jpg','.h5').replace('images','ground_truth')
        #gt_file = h5py.File(gt_path)
        #target = np.asarray(gt_file['density'])
        #target = torch.FloatTensor(target)
        #mask_target = Smooth_heaviside(target)
        #mask_target = mask_target.numpy()
        #mask_target = np.where(mask_target>0.6,1,0)
        #prediction = prediction*mask_target
        #prediction = np.max(prediction) + np.min(prediction) - prediction
        #plt.axis('off');
        #plt.imshow(prediction)
        #plt.savefig("prediction1.jpg",bbox_inches='tight', pad_inches=0)
        
        #print(prediction.shape,np.max(prediction),np.min(prediction))
        #prediction = np.where(prediction<0,0,prediction)
        #prediction = (prediction-np.min(prediction))/(np.max(prediction)-np.min(prediction))
        
        mat = io.loadmat(img_name.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_'))
        gt = mat["image_info"][0,0][0,0][0]        
        k = np.zeros((img.shape[0],img.shape[1]))    
        depth = (depth-np.min(depth))/(np.max(depth)-np.min(depth))  # 范围变化至0～1之间
        depth = np.max(depth)+np.min(depth)-depth  # 越深，人头越小
        
        gt = mat["image_info"][0,0][0,0][0]
        for i in range(0,len(gt)):
            if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
                k[int(gt[i][1]),int(gt[i][0])]=1
        
        
        k = gaussian_filter_density(k,depth)

        with h5py.File(img_name.replace('.jpg','.h5').replace('images','depth_density_map'), 'w') as hf:
            hf['density'] = k
        
        # output
        # filename = os.path.join(
        #    output_path, os.path.splitext(os.path.basename(img_name))[0]
        # )
        # utils.write_depth(filename, prediction, bits=2)

        filename = os.path.join(
           output_path, os.path.splitext(os.path.basename(img_name))[0]
        )
        utils.write_depth(filename, depth, bits=2)

    print("finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path', 
        default=r"D:\renqun\share_newdas\das\shanghai\part_A_final\test_data\images",
        help='folder with input images'
    )

    parser.add_argument('-o', '--output_path', 
        default=r'D:\renqun\share_newdas\das\shanghai\part_A_final\test_data\depth',
        help='folder for output images'
    )

    parser.add_argument('-m', '--model_weights', 
        default=r"D:\renqun\new_das\new_das\das\csrnet_depth\dpt_large-midas-2f21e586.pt",
        help='path to the trained weights of model'
    )

    parser.add_argument('-t', '--model_type', 
        default='dpt_large',
        help='model type: dpt_large, dpt_hybrid, midas_v21_large or midas_v21_small'
    )

    parser.add_argument('--optimize', dest='optimize', action='store_true')
    parser.add_argument('--no-optimize', dest='optimize', action='store_false')
    # parser.set_defaults(optimize=True)
    parser.set_defaults(optimize=False)

    args = parser.parse_args()

    # default_models = {
    #     "midas_v21_small": "weights/midas_v21_small-70d6b9c8.pt",
    #     "midas_v21": "weights/midas_v21-f6b98070.pt",
    #     "dpt_large": "weights/dpt_large-midas-2f21e586.pt",
    #     "dpt_hybrid": "weights/dpt_hybrid-midas-501f0c75.pt",
    # }

    #if args.model_weights is None:
        #args.model_weights = default_models[args.model_type]

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # compute depth maps
    run(args.input_path, args.output_path, args.model_weights, args.model_type, args.optimize)
