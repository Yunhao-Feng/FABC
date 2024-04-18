"""导包"""
import torch
import numpy as np
import time
import argparse
import datetime
import yaml
import visdom

"""从其他文件中载入变量、函数和类"""
import config
from image_helper import ImageHelper



if __name__ == '__main__':  # 当前文件作为主程序时才会运行
    print("Start Training")  # 开始训练  
    np.random.seed(2024)  # 设定随机数种子
    time_start_load_everything = time.time()  # 得到当前时间
    parser = argparse.ArgumentParser(description='PPDL')  # 设定参数解析器
    parser.add_argument('--params', dest='params')  # 添加参数
    args = parser.parse_args()  # 解析参数
    with open(f'./{args.params}', 'r') as f:
        params_loaded = yaml.load(f)  # 从yaml文件中解析参数
    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')  # 规整过的时间
    
    """选择哪个数据集开展实验"""
    if params_loaded['type'] == config.TYPE_CIFAR:
        helper = ImageHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', 'cifar'))
        helper.load_data()
    elif params_loaded['type'] == config.TYPE_MNIST:
        helper = ImageHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', 'mnist'))
        helper.load_data()
    elif params_loaded['type'] == config.TYPE_TINYIMAGENET:
        helper = ImageHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', 'tiny'))
        helper.load_data()
    else:
        helper = None
    
    
    
    
    
    