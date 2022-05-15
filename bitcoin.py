import numpy
from dtaidistance import dtw
# from dtaidistance import dtw_visualisation as dtwvis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def step2(series1, series2):
    '''

    :param series1: 第一个序列
    :param series2: 第二个序列
    :param s1begin: 第一个时间序列开始数据（需为正整数）
    :param s1end: 第一个时间序列结束数据（需为正整数）
    :param d: 时滞大小
    :return:输出DTW距离
    '''
    series1 = numpy.array(series1, dtype=numpy.double)
    series2 = numpy.array(series2, dtype=numpy.double)
    # TODO 这边最好加一点判断机制：比如说想要读取的数据长度超过了本身序列长度
    # if s1begin >= s1end or s1begin >= len(series1) or s1end >= len(series1) or \
    #         (s1begin + d) >= len(series2) or (s1end + d) >= len(series2):
    #     print("step2函数报错：请核查读取索引是否超出序列范围！")
    #     return
    # 先根据时滞大小从原有序列提取出新的时序, 算头去尾
    # new_series = np.stack((series1[s1begin:s1end], series2[s1begin + d:s1end + d]))
    # print(new_series)
    # 绘制对应曲线
    # path = dtw.warping_path(new_series[0], new_series[1])
    # dtwvis.plot_warping(new_series[0], new_series[1], path, filename="warp.png")
    # 计算距离矩阵
    ds = dtw.distance(series1, series2)
    return ds
    # print(ds)


if __name__ == '__main__':
    print(step2([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0], 0, 3))
