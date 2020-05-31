
# -*- coding: utf-8 -*-
"""
绘制3d图形
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def plot_3d(np_3d):
    # np 为3维张量
    shape = np_3d.shape
    for i in range(shape[0]):
        # 定义figure
        fig = plt.figure()
        # 创建3d图形的两种方式
        # 将figure变为3d
        ax = Axes3D(fig)
        # 定义x, y
        x = np.arange(0, shape[2], 1)
        y = np.arange(0, shape[1], 1)

        # 生成网格数据
        X, Y = np.meshgrid(x, y)

        # 计算每个点对的长度
        Z = np_3d[i,:,:]  #.flatten()
        print(Z.shape)
        # 绘制3D曲面
        # rstride:行之间的跨度  cstride:列之间的跨度
        # rcount:设置间隔个数，默认50个，ccount:列的间隔个数  不能与上面两个参数同时出现
        # cmap是颜色映射表
        # from matplotlib import cm
        # ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, cmap = cm.coolwarm)
        # cmap = "rainbow" 亦可
        # 我的理解的 改变cmap参数可以控制三维曲面的颜色组合, 一般我们见到的三维曲面就是 rainbow 的
        # 你也可以修改 rainbow 为 coolwarm, 验证我的结论
        ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, cmap = plt.get_cmap('rainbow'))

        plt.show()

def plot_2d(np_2d):
    # np 为3维张量
    shape = np_2d.shape
    for i in range(shape[0]):
        plt.figure(i)  # 创建了一个figure对象;

        # figure对象的add_axes()可以在其中创建一个axes对象,
        # add_axes()的参数为一个形如[left, bottom, width, height]的列表,取值范围在0与1之间;
        # 我们把它放在了figure图形的上半部分，对应参数分别为：left, bottom, width, height;
        plt.plot(np_2d[i, :])
        plt.show()