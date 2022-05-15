import pandas as pd

1
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


if __name__ == "__main__":

    # df显示所有data
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)

    # 读取数据csv
    coin = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'EOSUSDT', 'LTCUSDT', 'XLMUSDT', 'ADAUSDT', 'IOTAUSDT']
    data_b = pd.read_csv(r'./DATASET/' + coin[0] + '/' + coin[0] + '.csv', encoding='gbk', parse_dates=True)

    # 取出比特币时间戳和收盘价格
    final_df = data_b[['Timestamp']]  # 事先将时间戳放入最终的df
    data_all = data_b[['Close']]

    for item in coin:
        data = pd.read_csv(r'./DATASET/' + item + '/' + item + '.csv', encoding='gbk', parse_dates=True)
        # data = normalization(data[['Close']])
        data = data[['Close']]
        final_df[item] = data
        # data_all = pd.concat([data_all, data_else], axis=0).reset_index(drop=True)
    all = final_df.iloc[:, 1:].to_numpy()
    all = all.reshape(-1,1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    all = scaler.fit_transform(all.reshape((-1, 1), order='F'))
    all = all.reshape((8, 4193)).T
    # print(final_df)
    final = pd.DataFrame(all, columns=coin)
    final.insert(0, 'Timestamp', final_df['Timestamp'])
    # 制作归一化后每个币种对应每个时刻的价格
    # for i in range(len(coin)):
    #     data_normal = normal_df[i * len(data_b):(i + 1) * len(data_b)]
    #     data_normal = data_normal.rename(columns={data_normal.columns[-1]: coin[i]}).reset_index(drop=True)

    # final.to_csv('./DATASET/Bitcoin_dataset_v3.csv', index=False)
    # print(final_df)

    # ****
    # # 归一化
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # normal_df = pd.concat([d1, d2], axis=0).reset_index(drop=True)
    # normal_df = pd.DataFrame(scaler.fit_transform(normal_df))
    # d1_norm = normal_df[0:len(d1)]
    # d1_norm = d1_norm.rename(columns={d1_norm.columns[-1]: 'BTC'}).reset_index(drop=True)
    # d2_norm = normal_df[len(d1):]
    # d2_norm = d2_norm.rename(columns={d2_norm.columns[-1]: 'ETH'}).reset_index(drop=True)
    #
    # 拼接
    # new = pd.concat([timestamp, d1_norm, d2_norm], axis=1)
    # print(new)
    #
    # # 取出数据，放入DTW中计算距离矩阵
    # s1 = new['BTC'].values
    # s2 = new['ETH'].values
    # bt.step2(s1, s2, 0, 1000, d=0)
