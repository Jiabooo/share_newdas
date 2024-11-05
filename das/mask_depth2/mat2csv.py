# encoding=UTF8
import pandas as pd
import scipy
from scipy import io
import os
#遍历文件夹
for dirname, _, filenames in os.walk('./data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        # print(filename)
        # print(os.path.realpath(filename))  # 获取当前文件路径
        print(os.path.dirname(os.path.realpath(filename)))  # 从当前文件路径中获取目录
        # print(os.path.basename(os.path.realpath(filename)))  # 获取文件名
        (file, ext) = os.path.splitext(os.path.realpath(filename))
        # print(file)
        print(os.path.basename(os.path.realpath(file)))  # 获取文件名
        # print(ext)
        print(dirname)


        path = os.path.join(dirname, filename)
        # 1、导入文件
        matfile = scipy.io.loadmat(path)
        # 2、加载数据
        datafile = list(matfile.values())[-1]
        # 3、构造一个表的数据结构，data为表中的数据
        dfdata = pd.DataFrame(data=datafile)
        # 4、保存为.csv格式的路径
        datapath = dirname+'\\'+os.path.basename(os.path.realpath(file))+'.csv'
        # 5、保存为.txt格式的路径
        dfdata.to_csv(datapath, index=False)


