import torch
import numpy as np
from torch.utils.data import Dataset  # 抽象类，无法实例化
from torch.utils.data import DataLoader  # 加载数据的
import torch.nn.functional as F


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # 这个和前面差不多

    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred


model = LogisticRegressionModel()
criterion = torch.nn.BCELoss(size_average=False)  # cross——entropy：交叉商
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 优化器


class DiabetesDataset(Dataset):  # 括号里面写这个表示要继承这个类（可以不写）
    """1:在init里面全部把不多的数据放进
    去，或者2：把文件名放在列表里，然后在getitem里面再读出来"""
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])  # 行从开头到结尾，列从开头到结尾但是结尾最后一行不要
        self.y_data = torch.from_numpy(xy[:, [-1]])  # 所有行，然后倒数第一列是要的，用中括号表示最后拿出来的是一个列·向量

    def __getitem__(self, index):  # 是将来实例化后能够支持索引操作（魔法方法）
        return self.x_data[index], self.y_data[index]  # 这个返回一个元组,x数据的第i行，
        # y数据的第i行，这个函数与DATALOADER和enumerate直接挂钩，这个函数return出什么，那么data里面就是什么

    def __len__(self):  # 魔法方法
        return self.len


dataset = DiabetesDataset('diabetes.csv.gz')  # 进行抽象函数实例化，这个函数可以做索引和求长度
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=2)  # 1传递数据集2小批量是多少3打乱4读取的时候多线程，要读几次全部读完，如果遇到报
# 错加一个"if name == 'main'"
for epoch in range(100):
    for i, data in enumerate(train_loader, 0):  # 或者直接把data变成（inputs，labels）
        # 1
        inputs, labels = data
        # 2
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        print(epoch, i, loss.item())
        # 3
        optimizer.zero_grad()
        loss.backward()
        # 4
        optimizer.step()



