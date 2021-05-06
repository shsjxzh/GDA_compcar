import os
from easydict import EasyDict
import matplotlib.pyplot as plt
import torch
import numpy as np
import random
import pickle


# set experiment configs
opt = EasyDict()

# batch size
opt.batch_size = 3

# normalize each data domain
opt.normalize_domain = False
opt.num_domain = 3

# print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))
# load the data
from dataset import *

# 这里开始，重新跟新下pickle tool等级

with open('../data/data.pickle', "rb") as datafile:
    data_pkl = pickle.load(datafile)

with open('../data/data2.pickle', 'wb') as datafile:
    pickle.dump(data_pkl, datafile, protocol=4)

# # print(f"Data: {data_pkl['data'].shape}\nLabel: {data_pkl['label'].shape}")

# data = np.zeros((2,9)) + np.arange(1, 10)
# data = np.transpose(data)
# # print(data)
# label = np.array([1, 0, 1, 1, 0, 1, 1, 0, 1])
# domain = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

# data_pkl = {'data':data, 'label':label, 'domain':domain}
# # build dataset
# # data = data_pkl['data']
# # data_mean = data.mean(0, keepdims=True)
# # data_std = data.std(0, keepdims=True)
# # data_pkl['data'] = (data - data_mean) / data_std  # normalize the raw data
# datasets = [ToyDataset(data_pkl, i, opt) for i in range(opt.num_domain)]  # sub dataset for each domain
# dataset = SeqToyDataset(datasets, size=len(datasets[0]))  # mix sub dataset to a large one
# dataloader = DataLoader(
#     dataset=dataset,
#     shuffle=True,
#     batch_size=opt.batch_size
# )

# G = torch.FloatTensor(
#         [
#             [1, 0],
#             [1, 1],
#             [0, 1],
#         ]
#     )

# # i = 0
# for data in dataloader:
#     print(data)
#     """
#     :param
#         x_seq: Number of domain x Batch size x Data dim
#         y_seq: Number of domain x Batch size x Label dim
#     """
#     x_seq, y_seq = [d[0][None, :, :] for d in data], [d[1][None, :] for d in data]
#     # onehot for t
#     t_seq = [torch.nn.functional.one_hot(d[2], 3) for d in data]
#     x_seq = torch.cat(x_seq, 0)
#     y_seq = torch.cat(y_seq, 0)
#     t_seq = torch.cat(t_seq, 0).reshape(3, opt.batch_size, -1)
#     # print(t_seq)
#     # print(t_seq.size())
#     # print(torch.matmul(t_seq.float(), G))
#     print(x_seq[0])
#     print(x_seq[1])
#     m = (x_seq[0] * x_seq[1]).sum(1)
#     # m = torch.dot(x_seq[0], x_seq[1])
#     print(m)
#     print(m.size())

#     break