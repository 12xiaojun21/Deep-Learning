import torch

x_data = torch.tensor([[1.0], [2.0], [3.0]])  # 这是表示列矩阵，行矩阵是【【1，2，3】】【】中的【】表示一行
y_data = torch.tensor([[2.0], [4.0], [6.0]])


class LinearModel(torch.nn.Module):
    """这是模版，这两个函数不能省"""

    def __init__(self):  # 构造函数
        super(LinearModel, self).__init__()  # 继承
        self.linear = torch.nn.Linear(1, 1)  # nn.Linear是pytorch里面
        # 的一个类，（1，1）表示构造一个对象linear，包含了权重和偏置这两个Tensor

    def forward(self, x):  # 就叫这个名字，前馈的计算
        y_pred = self.linear(x)
        return y_pred


model = LinearModel()
criterion = torch.nn.MSELoss(size_average=False)  # 用来算损失的
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 优化器

for epoch in range(500):
    y_pred = model(x_data)
    l = criterion(y_pred, y_data)
    print(epoch, l.item())

    optimizer.zero_grad()  # 这一步的作用是梯度的归0，在进行反向传播前，要梯度归零
    l.backward()
    optimizer.step()

print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

x_test = torch.Tensor([4.0])
y_test = model(x_test)
print('y_test = ', y_test.item())


