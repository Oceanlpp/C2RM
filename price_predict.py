import numpy
import torch
import torch.nn as nn
from model.RNN import RNNModel
from model.LSTM import LSTMModel
from model.GRU import GRUModel
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
import bitcoin as bt


def naive(dataX, look_back):
    dataX = dataX.cpu().numpy()
    dataY = []
    # dataY.append([np.mean(dataX[:, -1, :], axis=-1)] * (look_back // 8))
    dataY.append([dataX[:, -1, 0]] * (look_back // 8))
    dataY = np.array(dataY)
    dataY = dataY.transpose(2, 1, 0)

    return dataY


def window_maxmin(dataset, look_back=1):
    shape = dataset.shape
    n = len(dataset) // look_back
    n = n * look_back
    dataset = dataset.T
    r = []
    a, b = 0.5, 0.5
    for data in dataset:
        d0 = numpy.zeros_like(data[0:look_back])
        f0 = numpy.zeros_like(data[0:look_back])
        for i in range(0, n, look_back):
            s = data[i:(i + look_back)]
            dn = d0 - a * (d0 - ((max(s) + min(s)) / 2))
            fn = f0 - b * (f0 - max(s) + min(s))
            r.append((s - dn) / fn)
            d0 = dn
            f0 = fn
    dat = numpy.reshape(r, (shape[1], -1))
    return dat.T


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(0, len(dataset) - look_back - 1, look_back):
        # dataX.append(np.mean(dataset[i:(i + look_back), 1:], axis=-1))  # [B,timestamp,input_dim]
        dataX.append(dataset[i:(i + look_back), 0])  # [B,timestamp,input_dim]
        # dataY.append(dataset[(i + 1):(i + look_back + 1), 0])  # [B,timestamp,input_dim]
        dataY.append([dataset[i + look_back, 0]])  # [B,timestamp,output_dim]
    return np.array(dataX), np.array(dataY)


# 多对多，窗口移位=窗口大小
def create_dataset_multi(dataset, look_back=1, dtw=False, asyn=False, split=0):
    dataX, dataY = [], []
    i = 0
    while (i + look_back) < len(dataset):
        dX = dataset[i:(i + look_back), 1:]
        dY0 = dataset[i:(i + look_back), 0]
        dY = dataset[(i + look_back):(i + look_back + look_back // 8), 0]
        # dY = dataset[i + look_back, 0]
        if dtw:
            distance = []
            # scaler = StandardScaler()
            if not asyn:
                for j in range(7):
                    distance.append(np.log(bt.step2(dX[:, j], dY0)))  # 调用DTW进行计算
                dX = dX * distance  # 加权不聚合
            else:
                for j in range(7):
                    for k in range(0, look_back, look_back // split):
                        for l in range(0, look_back, look_back // split):
                            distance.append(
                                (np.log(
                                    bt.step2(dX[k:(k + look_back // split), j],
                                             dY0[l:(l + look_back // split)]))))  # 调用DTW进行计算
                distance = np.reshape(distance, (7, split, split))
                distance = np.triu(distance)
                distance = np.reshape(distance, (7, -1))
                # distance = np.mean(distance, keepdims=True)
                dX = np.matmul(dX, distance)  # 聚合

        dataX.append(dX)  # [B,timestamp,input_dim]
        dataY.append(dY)  # [B,timestamp,output_dim]
        # i += 1
        i += look_back
    return np.array(dataX), np.array(dataY)


def plotCurve(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None, legend=None, figsize=(3.5, 2.5)):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')

    if legend:
        plt.legend(legend)
    plt.show()


if __name__ == "__main__":
    # cuda initialize
    cuda = True if torch.cuda.is_available() else False
    # fix random seed for reproducibility
    torch.manual_seed(125)
    if cuda:
        torch.cuda.manual_seed_all(125)
        torch.cuda.set_device(0)

    ratio = 0.8
    input_dim = 36
    timestep = 24
    hidden_dim = 32
    output_dim = 1
    num_epochs = 2000
    learning_rate = 0.01
    patience = 100
    dtw = True
    asyn = True
    split = 6
    diff = False
    shuffle = False

    # preprocess
    print("creating dataset")
    # read price
    df = pd.read_csv("DATASET/Bitcoin_dataset_v3.csv", index_col=False)
    # read dtw_matrix
    coin = ['ETHUSDT', 'XRPUSDT', 'EOSUSDT', 'LTCUSDT', 'XLMUSDT', 'ADAUSDT', 'IOTAUSDT']
    # for item in coin:
    #     dtw.append(pd.read_csv("dtw_data/" + item + "_v2.csv"))
    dataset = df.iloc[:, 1:]  # 去除时间列
    dataset = np.reshape(dataset.values, (-1, 8))

    # 这一步构建差分数据
    if diff:
        dataset = np.diff(dataset, axis=0)

    # dataset = window_maxmin(dataset, timestep)
    # print(dataset.shape)
    # print(dataset)
    # split into train and test sets, 50% test data, 50% training data

    # scaler_other_train = MinMaxScaler()
    # scaler_bit_train = MinMaxScaler()
    # scaler_other_test = MinMaxScaler()
    # scaler_bit_test = MinMaxscaler()
    #
    # dataset_other_train = scaler_other_train.fit_transform(dataset[:, 1:])
    # dataset_bit_train = scaler_bit_train.fit_transform(dataset[:, 0].reshape(-1, 1))
    # # dataset = np.append(dataset_bit_train, dataset_other_train, axis=-1)

    train_size = int(len(dataset) * ratio)
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
    trainX, trainY = create_dataset_multi(train, timestep, dtw, asyn, split)
    testX, testY = create_dataset_multi(test, timestep, dtw, asyn, split)
    # 创建多对一
    # trainX, trainY = create_dataset(train, timestep)
    # testX, testY = create_dataset(test, timestep)

    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], timestep, -1))
    testX = np.reshape(testX, (testX.shape[0], timestep, -1))

    if cuda:
        trainX = torch.from_numpy(trainX).cuda().float()
        testX = torch.from_numpy(testX).cuda().float()
        trainY = torch.from_numpy(trainY).cuda().float()
        testY = torch.from_numpy(testY).cuda().float()
    else:
        trainX = torch.from_numpy(testX).float()
        testX = torch.from_numpy(testX).float()
        trainY = torch.from_numpy(trainY).float()
        testY = torch.from_numpy(testY).float()
    if shuffle:
        train_index = torch.randperm(trainX.shape[0])
        test_index = torch.randperm(testX.shape[0])
        trainX = trainX[train_index]
        trainY = trainY[train_index]
        testX = testX[test_index]
        testY = testY[test_index]
    print(trainX.shape)
    print(trainY.shape)
    print(testX.shape)
    print(testY.shape)

    # model
    print("initializing model")
    # model = LSTMModel(input_dim, hidden_dim, output_dim)
    model = GRUModel(input_dim, hidden_dim, output_dim)
    # model = RNNModel(input_dim, hidden_dim, output_dim)
    if cuda:
        model.cuda()
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train and test
    # summary(model, input_size=(timestep, input_dim))
    print("start training")
    train_ls = torch.zeros(num_epochs)
    test_ls = torch.zeros(num_epochs)
    best_loss = 999.0
    best_epoch = 0
    for epochs in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(trainX)
        loss = criterion(outputs, trainY)
        if cuda:
            loss.cuda()
        loss.backward()
        optimizer.step()

        train_ls[epochs] = loss

        model.eval()
        predict = model(testX)
        test_loss = criterion(predict, testY)
        test_ls[epochs] = test_loss

        if (epochs + 1) % 100 == 0:
            print("train_loss = %f, test_loss = %f" % (loss, test_loss))
        if test_loss < best_loss:
            best_loss = test_loss
            best_epoch = epochs
            torch.save(model.state_dict(), "./save_model.pkl")
    # 绘制原始数据的图
    # all_series = scaler.inverse_transform(dataset)
    # all_series = np.array(all_series.tolist())
    # df_index = df.iloc[:, 0].tolist()
    # plt.figure(figsize=(12, 8))
    # plt.plot(all_series, label='real-data')
    # plt.show()

    # predict
    model.load_state_dict(torch.load("./save_model.pkl"))
    model.eval()
    predict = model(testX)
    # predict = torch.squeeze(torch.from_numpy(naive(testX, timestep)))
    final_test_ls = criterion(predict, testY)

    print("\ntest loss = %f" % final_test_ls)
    predict = np.reshape(predict.cpu().detach().numpy(), (-1, 1))
    # predict = scaler_bit_train.inverse_transform(predict)
    testY = np.reshape(testY.cpu().detach().numpy(), (-1, 1))
    # testY = scaler_bit_train.inverse_transform(testY)
    # 绘图
    print(best_epoch)
    result = np.array(predict)
    result.tofile('result/121_GRU.csv', sep=',', format='%f')
    plt.figure(figsize=(12, 8))
    plt.plot(np.ravel(testY[:, -1]), 'r', label='test')
    plt.plot(np.ravel(predict[:, -1]), 'b', label='predict')
    plt.legend()
    plt.show()
    plotCurve(range(1, num_epochs + 1), train_ls.tolist(),
              "epoch", "loss",
              range(1, num_epochs + 1), test_ls.tolist(),
              ["train", "test"]
              )
