import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w1 = torch.Tensor([0.1])
w2 = torch.Tensor([1.0])
b = torch.Tensor([0.1])
w1.requires_grad = True
w2.requires_grad = True
b.requires_grad = True


def forward(x):
    return w1 * (x * x) + w2 * x + b


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


for epoch in range(5000):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()  # 这个会算w1，w2，b的梯度1吧嘻嘻
        print('\tgrad :', x, y, 'w1:', w1.grad.item(), 'w2:', w2.grad.item(), 'b:', b.grad.item())
        w1.data = w1.data - w1.grad.data * 0.01
        w2.data = w2.data - w2.grad.data * 0.01
        b.data = b.data - b.grad.data * 0.01
        w1.grad.data.zero_()
        w2.grad.data.zero_()
        b.grad.data.zero_()
    print('progress:', epoch, l.item())
print('forward(4):', forward(4).item())


