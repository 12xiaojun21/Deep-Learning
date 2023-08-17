import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0


def forward(x):
    return x * w


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


def sgrad(x, y):
    return 2 * x * (x * w - y)


for epoch in range(100):
    for x, y in zip(x_data, y_data):
        grad_val = sgrad(x, y)  # 对每一个样本求梯度
        print(f'Epoch = {epoch}, loss = {grad_val}, w = {w}')
        w -= 0.01 * grad_val  # 拿一个样本就去更新
