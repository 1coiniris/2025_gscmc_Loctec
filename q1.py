import function.data_load as data_fun
import function.function as fun
import torch
import ast
import numpy as np
from scipy.io import savemat, loadmat
import argparse
import time

train_filename = ['data/data_process/341c_train.mat',
            'data/data_process/bc98_train.mat',
            'data/data_process/d0c1_train.mat']
test_filename = ['data/341c_val.mat',
            'data/bc98_val.mat',
            'data/d0c1_val.mat']

train_map_list = []
for data_file in train_filename:
    data = loadmat(data_file)
    train_map_list.append(data)

map_341c_train = train_map_list[0]
map_bc98_train = train_map_list[1]
map_d0c1_train = train_map_list[2]

# 聚合所有样本（便于全局拟合 α/β）
Gamma_all, Rate_all, file_marks = [], [], []

per_file_results = []

arrays = []
N = len(map_341c_train['csi_state'])
# 按行和列的顺序收集所有数组
for row in range(2):  # 行索引从0到2
    for col in range(4):  # 列索引从0到3
        array_name = map_341c_train[f"csi_matrix_r{row}_c{col}"]
        arrays.append(array_name)
# 将所有数组合并成一个形状为(8, 7000, 122)的数组
combined = np.stack(arrays, axis=0)
H = combined.transpose(1, 2, 0).reshape(N, 122, 2, 4)

SINR_i = np.zeros([N,122])
for i in range(N):
    for j,h in enumerate(H[i]):
        h = np.array(h)
        h_H = np.array(h).conjugate().T
        power = 10*np.log10(pow(np.linalg.norm(h, 'fro'),2)*1000)
        SINR_i[i,j] = power - map_341c_train['noise_floor'][i]


# 拼接 gamma_sc 与 rate（只保留有标签的文件）
Gamma_with_rate = [Gamma_all[i] for i, rate in enumerate(Rate_all) if rate is not None]
Rate_with_label = [rate for rate in Rate_all if rate is not None]

a = 1