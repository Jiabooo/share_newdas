import scipy.io as io
from matplotlib import pyplot as plt
import numpy as np
import cv2
import h5py
import os
import glob
#import Image
from matplotlib import cm as CM
import scipy

# 高斯核
def gaussian_filter_density(gt):
    print(gt.shape)
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))

    # 构造KDTree寻找相邻的人头位置
    leafsize = 2048
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    distances, locations = tree.query(pts, k=4)

    print ('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            # 构造三个人头的平均距离，其中beta=0.3
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
            if sigma > 5:
                sigma = 5
        else:
            sigma = np.average(np.array(gt.shape))/2./2. # case: 1 point
        # sigma = 10
        density += scipy.ndimage.gaussian_filter(pt2d, sigma, mode='constant')
    print ('done.')
    return density

# # set the root to the Shanghai dataset you download
root = r'G:\renqun\das\das\shanghai'

# now generate the ShanghaiA's ground truth
part_A_train = os.path.join(root,'part_A_final/train_data','images')
part_A_test = os.path.join(root,'part_A_final/test_data','images')
part_B_train = os.path.join(root,'part_B_final/train_data','images')
part_B_test = os.path.join(root,'part_B_final/test_data','images')
path_sets = [part_A_train,part_A_test]


img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

for img_path in img_paths:
    print(img_path)
    mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_'))
    img= plt.imread(img_path)
    k = np.zeros((img.shape[0],img.shape[1]))
    gt = mat["image_info"][0,0][0,0][0]
    for i in range(0,len(gt)):
        if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
            k[int(gt[i][1]),int(gt[i][0])]=1
    k = gaussian_filter_density(k)
    with h5py.File(img_path.replace('.jpg','.h5').replace('images','new_gt'), 'w') as hf:
            hf['density'] = k

# now see a sample from ShanghaiA
# for img_path in img_paths:
#     img_path = '/Users/luxixi/Desktop/CSRNet-pytorch-master/shanghai_data/part_A_final/train_data/images/IMG_' + str(
#         img_path) + '.jpg'
#     gt_file = h5py.File(img_path.replace('.jpg', '.h5').replace('images', 'sigma_10'), 'r')
#     groundtruth = np.asarray(gt_file['density'])
#     print(int(np.sum(groundtruth)))
#     heatmapshow = None
#     heatmapshow = cv2.normalize(groundtruth, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#     heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
#     # Output =  'Sigma5_A117.bmp'
#     # cv2.imwrite(Output, heatmapshow)
#     cv2.imshow("Heatmap", heatmapshow)
#     cv2.waitKey(0)

# imgs = [20, 181, 89]
# for img in imgs:
#     img_path = '/Users/luxixi/Desktop/CSRNet-pytorch-master/shanghai_data/part_A_final/test_data/images/IMG_' + str(img) + '.jpg'
    # print(img_path)
    # gt_file = h5py.File(img_path.replace('.jpg', '.h5').replace('images', 'ground_truth'), 'r')
    # groundtruth = np.asarray(gt_file['density'])
    # print(int(np.sum(groundtruth)))
