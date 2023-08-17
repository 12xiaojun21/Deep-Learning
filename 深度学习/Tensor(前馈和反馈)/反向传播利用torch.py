import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.Tensor([1.0])  # 创建权重，别忘记了中括号，即使只有一个
w.requires_grad = True  # 要求其计算梯度


def forward(x):
    return x * w  # w在这里是权重的意思，
    # 这里的乘法的性质也改变了，x也会被类型转换为Tensor，
    # 然后得出的结果也是一个Tensor


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)  # 前馈，l也是一个tensor，后面需要用l的值可以用l.item（）
        l.backward()  # 这条反向会把这一条链上所有需要求梯度的都求出来
        # ，做完这个运算后计算图也就没有了
        print('\tgrad :', x, y, w.grad.item())  # 防止来产生计算图，把梯度里面的数值直接拿出来，变成标量
        w.data = w.data - w.grad.data * 0.01  # 不能用w.grad直接
        # 乘，因为那样会再进行一次类型转换然后再次绘图
        w.grad.data.zero_()  # 把权重里面梯度的数据全部清零，不清的话每次的梯度就会累加。
    print("progress:", epoch, l.item())
print(forward(4).item())

