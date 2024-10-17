# coding=gbk
from swnms_utils_coord import *
from swnms_utils_peak import *

def test_ensure_spacing():
    # 创建一个示例的动态间距数组
    distance_array = np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 2, 1, 1]
    ])

    # 创建一个示例的坐标数组
    pic = np.array([
        [0, 0, 0, 0],
        [0, 1, 2, 0],
        [1, 1, 0, 0],
        [1, 2, 0, 0],
    ])


    # 测试 max_out 限制
    coordinates = peak_local_max(pic, distance_array=distance_array,num_peaks=3,exclude_border=False)
    print("Result:", coordinates)


test_ensure_spacing()