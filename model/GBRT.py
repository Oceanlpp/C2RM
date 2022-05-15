import torch
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
import pandas as pd
import torch.nn as nn
import datetime
import collections
from matplotlib import pyplot as plt

# plt.style.use('ggplot')
import seaborn as sns
import matplotlib.dates as mdates
# from pandas.plotting import register_matplotlib_converters

# register_matplotlib_converters()
# pd.options.mode.chained_assignment = None

import lightgbm
import xgboost as xgb
import shap


# shap.initjs()


def step_series(n, mean, scale, n_steps):
    s = np.zeros(n)
    step_idx = np.random.randint(0, n, n_steps)
    value = mean
    for t in range(n):
        s[t] = value
        if t in step_idx:
            value = mean + scale * np.random.randn()
    return s


def linear_link(x):
    return x


def sigmoid_link(x, scale=10):
    return 1 / (1 + np.exp(-scale * x))


def mem_link(x, length=50):
    mfilter = np.exp(np.linspace(-10, 0, length))
    return np.convolve(x, mfilter / np.sum(mfilter), mode='same')


def create_signal(links=[linear_link, sigmoid_link, mem_link]):
    days_year = 365
    quaters_year = 4
    days_week = 7

    # two years of data, daily resolution
    idx = pd.date_range(start='2018-01-01', end='2020-01-01', freq='D')

    df = pd.DataFrame(index=idx, dtype=float)
    df = df.fillna(0.0)

    n = len(df.index)
    trend = np.zeros(n)
    seasonality = np.zeros(n)
    for t in range(n):
        trend[t] = 2.0 * t / n
        seasonality[t] = 2.0 * np.sin(np.pi * t / days_year * quaters_year)

    covariates = [step_series(n, 0, 2.0, 80), step_series(n, 0, 1.0, 80), step_series(n, 0, 1.0, 80)]
    covariate_links = [links[i](covariates[i]) for i in range(3)]

    noise = np.random.randn(n)

    signal = trend + seasonality + np.sum(covariate_links, axis=0) + noise

    df['signal'], df['trend'], df['seasonality'], df['noise'] = signal, trend, seasonality, noise
    for i in range(3):
        df[f'covariate_0{i + 1}'] = covariates[i]
        df[f'covariate_0{i + 1}_link'] = covariate_links[i]

    return df


#
# engineer features for the model
#
def features_regression(df):
    observed_features = ['covariate_01', 'covariate_02', 'covariate_03']
    dff = df[['signal'] + observed_features]

    dff['year'] = dff.index.year
    dff['month'] = dff.index.month
    dff['day_of_year'] = dff.index.dayofyear

    feature_lags = [7, 14, 21, 28, 35, 42, 49, 120, 182, 365]
    for lag in feature_lags:
        dff.loc[:, f'signal_lag_{lag}'] = dff['signal'].shift(periods=lag, fill_value=0).values

    return dff


#
# train-test split
#
def split_train_test(df, train_ratio):
    y_train, y_test = [], []
    x_train, x_test = [], []
    split_t = int(len(df) * train_ratio)

    y = df['signal']
    y_train = y[:split_t]
    y_test = y[split_t:]

    xdf = df.drop('signal', inplace=False, axis=1)
    x_train = xdf[:split_t]
    x_test = xdf[split_t:]

    return x_train, y_train, x_test, y_test


#
# fit LightGBM model
#
def fit_lightgbm(x_train, y_train, x_test, y_test, n_estimators=100, verbose_eval=50):
    model = lightgbm.LGBMRegressor(
        boosting_type='gbdt',
        # num_leaves = 8 - 1,
        n_estimators=n_estimators)

    model.fit(x_train,
              y_train,
              eval_set=[(x_train, y_train), (x_test, y_test)],
              eval_metric='mape',
              verbose=verbose_eval)

    return model


def fit_xgboost(x_train, y_train, x_test, y_test, n_estimators=20, verbose_eval=50):
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror",
                                 booster='gbtree',
                                 n_estimators=n_estimators)
    xgb_model.fit(x_train,
                  y_train,
                  eval_set=[(x_train, y_train), (x_test, y_test)],
                  eval_metric='rmse',
                  verbose=verbose_eval)

    return xgb_model


def plot_xgboost_metric(model):
    results = model.evals_result()
    x_axis = range(0, len(results['validation_0']['rmse']))
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
    ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
    ax.legend()


def create_dataset_multi(dataset, look_back=1):
    dataX, dataY = [], []
    i = 0
    while (i + look_back) < len(dataset):
        dX = dataset[i:(i + look_back), 1:]
        dY0 = dataset[i:(i + look_back), 0]
        dY = dataset[(i + look_back):(i + look_back + look_back // 8), 0]
        # dY = dataset[i + look_back, 0]
        dataX.append(dX)  # [B,timestamp,input_dim]
        dataY.append(dY)  # [B,timestamp,output_dim]'
        # i += 1
        i += look_back
    return np.array(dataX), np.array(dataY)


#
# generate data sample and fit the model
if __name__ == "__main__":
    input_dim = 7
    timestep = 24
    output_dim = 3
    diff = False
    shuffle = False

    # preprocess
    print("creating dataset")
    # read price
    df = pd.read_csv("../DATASET/Bitcoin_dataset_cong.csv", index_col=False)
    # read dtw_matrix
    # coin = ['ETHUSDT', 'XRPUSDT', 'EOSUSDT', 'LTCUSDT', 'XLMUSDT', 'ADAUSDT', 'IOTAUSDT']
    # for item in coin:
    #     dtw.append(pd.read_csv("dtw_data/" + item + "_v2.csv"))
    dataset = df.iloc[:, 1:]  # 去除时间列
    dataset = np.reshape(dataset.values, (-1, 8))

    # 这一步构建差分数据
    if diff:
        dataset = np.diff(dataset, axis=0)

    train_size = int(len(dataset) * 0.9)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    # dataset_other_train = scaler_other_train.fit_transform(train[:, 1:])
    # dataset_bit_train = scaler_bit_train.fit_transform(train[:, 0].reshape(-1, 1))  # 单独拿出来用于之后的反归一化
    # dataset_other_test = scaler_other_train.transform(test)
    # dataset_bit_test = scaler_bit_train.transform(test[:, 0].reshape(-1, 1))  # 单独拿出来用于之后的反归一化
    # #
    # train = np.append(dataset_bit_train, dataset_other_train, axis=-1)
    # test = np.append(dataset_bit_train, dataset_other_test, axis=-1)
    # # reshape into X=t and Y=t+1~2t, timestep 64
    # 创建多对多
    trainX, trainY = create_dataset_multi(train, timestep)
    testX, testY = create_dataset_multi(test, timestep)

    trainX = np.reshape(trainX, (trainX.shape[0], -1))
    testX = np.reshape(testX, (testX.shape[0], -1))
    # df = create_signal()
    # fig, ax = plt.subplots(len(df.columns), figsize=(20, 15))
    # for i, c in enumerate(df.columns):
    #     ax[i].plot(df.index, df[c])
    #     ax[i].set_title(c)

    # plt.tight_layout()
    # plt.show()
    # train_ratio = 0.8
    # x_train, y_train, x_test, y_test = split_train_test(df, train_ratio)
    other_params = {'learning_rate': 0.01, 'n_estimators': 200, 'max_depth': 6, 'min_child_weight': 1, 'seed': 125,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0.9, 'reg_lambda': 1,
                    'verbosity': 1}
    multioutputregressor = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:squarederror', **other_params)).fit(
        trainX, trainY)
    criterion = nn.MSELoss(reduction='mean')
    forecast = multioutputregressor.predict(testX)
    predict = torch.from_numpy(forecast).float()
    tensor_testY = torch.from_numpy(testY).float()
    print("%f" % criterion(predict, tensor_testY))

    # model = fit_lightgbm(trainX, trainY, testX, testY)  # can use fit_xgboost as an alternative

    #
    # plot the fitting metrics
    #
    # lightgbm.plot_metric(model, metric='mse', figsize=(10, 3))

    #
    # plot the forecast
    #
    # forecast = model.predict(testX)
    # result = np.array(predict)
    # result.tofile('../result/result_GBRT.csv', sep=',', format='%f')
    plt.figure(figsize=(12, 8))
    plt.plot(np.ravel(testY[:, -1]), 'r', label='test')
    plt.plot(np.ravel(forecast[:, -1]), 'b', label='predict')
    plt.legend()
    plt.show()
    # fig, ax = plt.subplots(1, figsize=(20, 5))
    # ax.plot(df.index, forecast, label='Forecast (7 days ahead)')
    # ax.plot(df.index, df['signal'], label='Actuals')
    # ax.axvline(x=df.index[int(len(df) * train_ratio)], linestyle='--')
    # ax.legend()
    # plt.show()
