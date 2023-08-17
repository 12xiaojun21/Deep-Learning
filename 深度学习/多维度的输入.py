import numpy as np
import torch

xy = np.loadtxt('diabetes.csv.gz', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])  # 行从开头到结尾，列从开头到结尾但是结尾最后一行不要
y_data = torch.from_numpy(xy[:, [-1]])  # 所有行，然后倒数第一列是要的，用中括号表示最后拿出来的是一个列·向量


