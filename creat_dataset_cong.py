import pandas as pd
import bitcoin as bt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# df显示所有data
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

# 读取数据csv
coin = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'EOSUSDT', 'LTCUSDT', 'XLMUSDT', 'ADAUSDT', 'IOTAUSDT']
data_b = pd.read_csv(r'./DATASET/' + coin[0] + '/' + coin[0] + '.csv', encoding='gbk', parse_dates=True)

# 取出比特币时间戳和收盘价格
final_df = data_b[['Timestamp']]  # 事先将时间戳放入最终的df
data_all = data_b[['Close']]  # TODO 这一步和上面那一步的读取csv可以写入下面的循环，但是无伤大雅，懒得精炼了

# 将其它币种的收盘价格以行为单位拼接进来，方便下一步的归一化
for i in range(len(coin) - 1):
    data_else = pd.read_csv(r'./DATASET/' + coin[i + 1] + '/' + coin[i + 1] + '.csv', encoding='gbk', parse_dates=True)
    data_else = data_else[['Close']]
    data_all = pd.concat([data_all, data_else], axis=0).reset_index(drop=True)

# 归一化
scaler = MinMaxScaler(feature_range=(0, 1))
normal_df = pd.DataFrame(scaler.fit_transform(data_all))

# 制作归一化后每个币种对应每个时刻的价格
for i in range(len(coin)):
    data_normal = normal_df[i * len(data_b):(i + 1) * len(data_b)]
    data_normal = data_normal.rename(columns={data_normal.columns[-1]: coin[i]}).reset_index(
        drop=True)  # reset_index的主要目的是防止拼接的时候自动使用索引进行拼接
    final_df = pd.concat([final_df, data_normal], axis=1)

final_df.to_csv('./DATASET/Bitcoin_dataset_cong.csv', index=False)
print(final_df)
