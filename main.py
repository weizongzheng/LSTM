import pandas as pd
import numpy as np
import torch
from torch import nn, optim


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.lin = nn.Linear(in_features=8, out_features=1)

    def forward(self, input):
        out, h = self.lstm(input)
        r = out[-1, -1]
        result = self.lin(r)
        return result


def getData():
    print("-----------读入数据--------------")
    data = pd.read_excel("./J00170.xlsx")
    data = data.iloc[:, 2]
    length = len(data)
    data = torch.tensor(data)
    print("-----------读入成功--------------")
    return data.to(torch.float32), length


def train():
    data, length = getData()
    train_size = int(length * 0.75)
    trainData = data[:train_size + 1]
    testData = data[train_size + 1:length + 1]
    test_size = len(testData)

    print("-----------模型训练--------------")
    EPOCH = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTM(1, 8, 2).to(device)
    criteon = torch.nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for i in range(EPOCH):
        print("开始进行第{}次训练".format(i))
        model.train()
        sum_train_loss = 0
        for j in range(1, train_size - 1):
            in_data = trainData[:j]
            if j > 25:
                in_data = in_data[-25:]
                in_data = in_data.view(1, 25, 1)
            else:
                in_data = in_data.view(1, j, 1)
            result = model(in_data)
            loss = criteon(result, trainData[j])

            sum_train_loss = sum_train_loss + loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if j % 200 == 0:
                print(loss.item())
        print("第{}次训练结束".format(i))
        print(i, sum_train_loss)

    print("---------开始进行预测---------")
    model.eval()
    x = []
    with torch.no_grad():
        for j in range(1, test_size - 1):
            in_data = testData[:j]
            if j > 25:
                in_data = in_data[-25:]
                in_data = in_data.view(1, 25, 1)
            else:
                in_data = in_data.view(1, j, 1)
            result = model(in_data)
            x.append(result.item())
    d = pd.DataFrame(x)
    d.to_excel('./result.xls')


if __name__ == '__main__':
    train()
