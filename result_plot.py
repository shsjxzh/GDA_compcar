import os
from easydict import EasyDict
import matplotlib.pyplot as plt
import torch
import numpy as np
import random
import pickle

from utils.utils import *

# data_source = 'data/data.pickle'
# data_source = 'data/toy_d15_quarter.pkl'

# TODO: deal with dataset that has different size!!!
data_source = 'data/quarter-circle.pkl'

from dataset_utils.dataset import *
with open(data_source, "rb") as datafile:
    data_pkl = pickle.load(datafile)
print(f"Data: {data_pkl['data'].shape}\nLabel: {data_pkl['label'].shape}")

# build dataset
data = data_pkl['data']
data_mean = data.mean(0, keepdims=True)
data_std = data.std(0, keepdims=True)


# import the experiment setting
from configs.config_quarter import opt
from utils.plot import plot_data_and_label
info = read_pickle(opt.outf + '/pred.pkl')
fig, ax = plt.subplots(1, 1, figsize=(6, 6 * 0.9))
print(info['data'].shape)
info['data'] = info['data'] * data_std + data_mean
plot_data_and_label(ax, info['data'], info['label'])
plt.title(f"Model's prediction: ", fontsize=20)
plt.show()