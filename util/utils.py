# -*- ecoding: utf-8 -*-
# @ProjectName: python-classifier-2022-master
# @ModuleName: util
# @Function: 
# @Author: Eliysiumar
# @Time: 2022/4/19 14:44

import matplotlib.pyplot as plt
import time
import math
import numpy as np
import os

def plot_data(data_list, pic_name = None,Fs=None, is_save=True, is_show = False):
    if pic_name is None:
        pic_name = str(time.time())
    plt.figure()
    img_num = len(data_list)
    row_num = int(math.sqrt(img_num))
    line_num = math.ceil(img_num/row_num)
    for i in range(line_num):
        for j in range(row_num):
            plt.subplot(line_num, row_num, i*row_num+j+1)
            data = data_list[i*row_num+j]
            if not Fs is None:
                if Fs is list:
                    fs = Fs[i*row_num+j]
                else:
                    fs = Fs
                x = np.linspace(0,data.shape[0]/fs, len(data))
            else:
                x = np.linspace(0,len(data)-1, len(data))
            plt.plot(x,data)
    plt.title(pic_name)
    if is_show:
        plt.show()
    if is_save:
        plt.savefig(pic_name + ".png")
    plt.close()

def check_dir(dir_path):
    """
    检查文件夹，若不存在则递归创建

    :param dir_path:
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

def drawPic(data,  # 要绘制的数据
            Fs=None,  # 采样率
            title=None,  # 标题
            is_show=False,  # 是否显示
            to_file=None,  # 输出图片位置
            highlight_point = None, # 高亮的点x轴
            extra_data = None, # 额外需绘制的线
            fig_size = None
            ):
    plt.figure(figsize=fig_size)
    if Fs :
        x = np.linspace(0, data.shape[0] / Fs, len(data))
    else:
        x = np.linspace(0, len(data) - 1, len(data))
    plt.plot(x,data)
    if not highlight_point is None:
        plt.plot(highlight_point, data[highlight_point], 'bx')
    if not extra_data is None:
        point_pattern = ['bx', 'bo', 'rx', 'ro']
        for idx in range(len(extra_data[0])):
            if not  int(extra_data[1][idx]) == 0:
                plt.plot(extra_data[0][idx], data[int(extra_data[0][idx]*Fs)], point_pattern[int(extra_data[1][idx]-1)])

    if not title is None:
        plt.title(title)
    if not to_file is None:
        plt.savefig(to_file)
    if is_show:
        plt.show()
    plt.close()