import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])  # y的分类


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

for epoch in range(1000):
    y_pred = model(x_data)
    l = criterion(y_pred, y_data)
    print(epoch, l.item())

    optimizer.zero_grad()  # 这一步的作用是梯度的归0，在进行反向传播前，要梯度归零
    l.backward()
    optimizer.step()
x = np.linspace(0, 10, 200)
x_t = torch.Tensor(x).view((200, 1))
y_t = model(x_t)
y = y_t.data.numpy()
plt.plot(x, y)
plt.plot([0, 10], [0.5, 0.5], c='r')
plt.xlabel("hours")
plt.ylabel('FF')
plt.grid()
plt.show()