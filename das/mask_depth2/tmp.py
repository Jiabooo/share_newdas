import numpy as np

import os


output_npy_dir = "threshold_res_npy/"
pic_num = 1


ar_load = np.load(output_npy_dir+'IMG_{}.npy'.format(pic_num))
print(ar_load)
