import numpy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x_data = [1.0, 2.0, 3.0]
y_data = [3.0, 5.0, 7.0]


def forward(x, w, b):
    return x * w + b


def loss(x, w, b, y):
    y_pred = forward(x, w, b)
    return (y - y_pred) * (y - y_pred)  # 求损失的嘻嘻


w_list = []
b_list = []
mse_list = []
for w in np.arange(0.0, 4.1, 0.1):
    for b in np.arange(0.0, 1.6, 0.1):
        print(f'w = {w}, b = {b}')
        L_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            y_val_pred = forward(x_val, w, b)
            L_sum += loss(x_val, w, b, y_val)
            print('\t', x_val, y_val, y_val_pred, loss(x_val, w, b, y_val))
        MSE = L_sum / 3
        print(f'MSE = {MSE}')
        w_list.append(w)
        b_list.append(b)
        mse_list.append(MSE)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(w_list, b_list, mse_list)
plt.show()
