#!/bin/bash
#指定作业提交的队列
#SBATCH -p hebhdnormal
#指定作业申请的节点数
#SBATCH -N 1
#指定每个节点运行进程数。
#SBATCH --ntasks-per-node=8
#指定任务需要的处理器数目
#SBATCH --cpus-per-task=1
#指定每个节点使用通用资源的名称及数量
#SBATCH --gres=dcu:2
#作业名称，使用squeue看到的作业名
#SBATCH -J first_test

#指定作业标准结果输出文件名称
#SBATCH -o %x.o%j
#指定作业标准错误输出文件名称
#SBATCH -e %x.e%j

source ~/miniconda3/etc/profile.d/conda.sh  #激活conda命令
conda activate jia     #激活自定义的虚拟环境


#运行环境
module load compiler/dtk/23.04 #加载对应安装torch的dtk模块

#运行程序
python addc_train.py A_train.json A_test.json 0 0