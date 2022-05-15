import pandas as pd
import bitcoin as bt
import numpy as np

# 定义时间序列分为几段
# TODO 可以修改这里来定义要将时间段分为几段
time_num = 10
# 定义币种名称
coin = ['ETHUSDT', 'XRPUSDT', 'EOSUSDT', 'LTCUSDT', 'XLMUSDT', 'ADAUSDT', 'IOTAUSDT']
# 创建距离字典，将币名与其对应的距离矩阵相联系起来
array_dic = {}
for item in coin:
    array_dic[item] = np.zeros((time_num, time_num))

# 读取csv，并且计算时间长度
df = pd.read_csv("DATASET/Bitcoin_dataset_v2.csv")
bitprice = df['BTCUSDT']
lenth = int(len(df) / time_num)

for item in coin:  # 第一层循环：对于每一种非比特币以外的其它币都需要计算他们对于比特币的影响
    for i in range(time_num):  # 矩阵应该是段数*段数，这里按行来更新，因此也需要循环和段数一样的次数
        # print(i)
        # TODO 抽取这个时刻的比特币的价格时序，这边可以做算法上的优化，可以将它事先保存起来，
        #  需要用于计算时可以直接提取，这样就不需要每次计算前提取一次了，
        #  但是这样也存在问题，就是要定义多个变量名，而且使用时也不是很方便。
        b_p = bitprice[lenth * i:lenth * (i + 1)]
        # print(b_p)
        for j in range(i + 1):  # 对于这个时刻的比特币，求其它之前时刻和现在时刻的币种对它的影响
            other_price = df[item]  # 首先提取出其它币种的全部价格曲线
            o_p = other_price[lenth * j:lenth * (j + 1)].values  # 抽出需要时间段的价格曲线，并将dataframe转换成array的形式
            distance = bt.step2(o_p, b_p, 0, lenth)  # 调用DTW进行计算
            # print(coin[x + 1])
            array_dic[item][i][j] = distance  # 在字典对应的的距离矩阵中填充距离
            # print(distance)
        print('*' * 10)
    df_new = pd.DataFrame(array_dic[item])
    df_new.to_csv('./dtw_data/' + item + '_v2.csv', index=False)
    print(array_dic[item])
