import pandas as pd
import numpy as np
import torch
# import io
from skimage import io
import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import  has_file_allowed_extension, is_image_file, IMG_EXTENSIONS, pil_loader, accimage_loader,default_loader
from PIL import Image
import cv2


# from data_loader.utils import get_date_list

def read_pickle(name):
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data

class CompcarsDataset(Dataset):
    def __init__(self, data_lists, domain_idx, sudo_len, transform, opt):
        self.domain_idx = domain_idx
        self.sudo_len = sudo_len
        # self.file_name = opt.data_src + opt.domain_name[domain_idx]
        self.data_list = data_lists[domain_idx] # pd.read_csv(self.file_name, header=None)
        self.real_len = len(self.data_list)
        self.transform = transform

    def __len__(self):
        # do not consider sudo len
        return self.sudo_len


    def __getitem__(self, idx):
        if idx >= self.real_len:
            idx = idx % self.real_len
        path, target = self.data_list[0][idx], self.data_list[2][idx]
        img = default_loader(path) # io.imread(path)
        # img = cv2.imread(path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)
        return img, target - 1, self.domain_idx


class CompcarsDataLoader():
    def __init__(self, transform, opt):
        self.opt = opt
        # these domains are given in number!!
        self.src_domain = opt.src_domain
        self.tgt_domain = opt.tgt_domain
        self.all_domain = opt.all_domain
        self.transform = transform

        self.data_lists = []
        self.sudo_len = 0
        for i in range(self.opt.num_domain):
            file_name = opt.data_src + opt.domain_name[i]
            data_list = pd.read_csv(file_name, header=None)
            self.sudo_len = max(len(data_list), self.sudo_len)
            self.data_lists.append(data_list)

        # data transform 依然没有解决
        self.train_datasets = [
            CompcarsDataset(
                self.data_lists,
                domain_idx=i,
                opt=opt,
                sudo_len=self.sudo_len,
                transform=self.transform
            ) for i in self.all_domain
        ]

        if self.opt.test_on_all_dmn:
            self.test_datasets = [
                CompcarsDataset(
                    self.data_lists,
                    domain_idx=i,
                    opt=opt,
                    sudo_len=self.sudo_len,
                    transform=self.transform
                ) for i in self.all_domain
            ]
        else:
            self.test_datasets = [
                CompcarsDataset(
                    self.data_lists,
                    domain_idx=i,
                    opt=opt,
                    sudo_len=self.sudo_len,
                    transform=self.transform
                ) for i in self.tgt_domain
            ]

        self.train_data_loader = [
            DataLoader(dataset, 
                batch_size=opt.batch_size, 
                shuffle=opt.shuffle,
                num_workers=2,
                pin_memory=True,
                ) 
                for dataset in self.train_datasets
        ]

        self.test_data_loader = [
            DataLoader(dataset, 
                batch_size=opt.batch_size, 
                shuffle=opt.shuffle,
                num_workers=2,
                pin_memory=True,
                ) 
                for dataset in self.test_datasets
        ]

    def get_train_data(self):
        # this is return a iterator for the whole dataset
        return zip(*self.train_data_loader)

    def get_test_data(self):
        return zip(*self.test_data_loader)


# class RegressDataLoader():
#     def __init__(self, opt):
#         self.opt = opt
#         self.src_domain = opt.src_domain
#         self.tgt_domain = opt.tgt_domain
#         self.raw_data = read_pickle(opt.data_src)
#         self.group_len = opt.group_len
#         self.all_states = self.opt.all_domain # self.src_domain + self.tgt_domain
#         self.sudo_len = max([len(self.raw_data[state]) for state in self.all_states]) - self.group_len + 1
#         self.all_mean, self.all_std = self.__norm__()
#         # self.sudo_len = max([len(self.raw_data[state]) for state in self.all_states]) - self.opt.seq_len
        
#         if self.opt.bound_prediction or self.opt.clamp_prediction:
#             self.max, self.min = self.__maxmin__()
#             self.norm_max, self.norm_min = (self.max - self.all_mean) / self.all_std, (self.min - self.all_mean) / self.all_std
# #             print(self.max)
# #             print(self.min)
        

#         self.train_datasets = [
#             CovidTrainDataset(self.raw_data[state], 
#                 isSrc=(state in self.src_domain), 
#                 all_mean=self.all_mean, 
#                 all_std=self.all_std,
#                 domain_idx=self.opt.state2num[state],
#                 sudo_len=self.sudo_len,
#                 opt=opt) 
#                 for state in self.all_states
#         ]

#         if self.opt.test_on_all_dmn:
#             # test on all domains
#             self.test_datasets = [
#                 CovidTestDataset(self.raw_data[state], 
#                     # isSrc=True, 
#                     all_mean=self.all_mean, 
#                     all_std=self.all_std,
#                     domain_idx=self.opt.state2num[state],
#                     # sudo_len=self.sudo_len,
#                     opt=opt) 
#                     for state in self.all_states
#             ]
#         else:
#             # test only on target domain
#             self.test_datasets = [
#                 CovidTestDataset(self.raw_data[state], 
#                     # isSrc=True, 
#                     all_mean=self.all_mean, 
#                     all_std=self.all_std,
#                     domain_idx=self.opt.state2num[state],
#                     # sudo_len=self.sudo_len,
#                     opt=opt) 
#                     for state in self.tgt_domain
#             ]
        
#         self.train_data_loader = [
#             DataLoader(dataset, 
#                 batch_size=opt.batch_size, 
#                 shuffle=opt.shuffle) 
#                 for dataset in self.train_datasets
#         ]

#         self.test_data_loader = [
#             DataLoader(dataset, 
#                 batch_size=opt.batch_size, 
#                 shuffle=opt.shuffle) 
#                 for dataset in self.test_datasets
#         ]
        
#     def __maxmin__(self):
#         tmp_max = -1e9
#         tmp_min = 1e9
#         for state in self.all_states:
#             tmp_max = max(tmp_max, max(self.raw_data[state]))
#             tmp_min = min(tmp_min, min(self.raw_data[state]))
        
#         return tmp_max, tmp_min

#     def __norm__(self):
#         ## TODO: complete the whole code
#         # be careful about the mean and variance calculation
#         seq_len = self.opt.seq_len
#         all_data = np.array([])
#         for state in self.src_domain:
#             # be sure that we delete more data (the last few days)
#             all_data = np.append(all_data, self.raw_data[state][:(self.sudo_len + seq_len - 1)])
#         for state in self.tgt_domain:
#             # for 244 days only, without many special cases!
#             tmp_data = self.raw_data[state]
#             # for i in range(int((tmp_data.shape[0] // seq_len) / 2)):
#             #     all_data = np.append(all_data, tmp_data[2 * seq_len * i: 2 * seq_len *i + 7])
#             # group_len = seq_len + 1
#             group_len = self.group_len
#             data_group_num = tmp_data.shape[0] // group_len
#             for i in range(data_group_num):
#                 all_data = np.append(all_data, tmp_data[i * group_len: i * group_len + seq_len])
#                 # all_data = np.append(all_data, tmp_data[i * group_len + group_len - 1])

#         return all_data.mean(), all_data.std()
#         # start here!


#     def get_train_data(self):
#         # this is return a iterator for the whole dataset
#         return zip(*self.train_data_loader)

#     def get_test_data(self):
#         return zip(*self.test_data_loader)


# class CovidTrainDataset(Dataset):
#     def __init__(self, data, isSrc, all_mean, all_std, domain_idx, sudo_len, opt):
#         self.data = (data - all_mean) / all_std
#         self.domain_idx = domain_idx
#         self.isSrc = isSrc
#         self.seq_len = opt.seq_len

#         self.sudo_len = sudo_len
#         self.opt = opt
#         self.group_len = opt.group_len

#         if isSrc:
#             # self.real_len = self.data.shape[0] - self.seq_len
#             self.real_len = self.data.shape[0] - self.opt.group_len + 1
#         else: 
#             # Assume that we have 244 days data, so doesn't handle many speciall cases!
#             # self.real_len = int((self.data.shape[0] // self.seq_len) / 2)
#             self.real_len = self.data.shape[0] // self.opt.group_len # (self.seq_len + 1)


#     def __len__(self):
#         return self.sudo_len

#     def __getitem__(self, idx):
#         if self.isSrc:
#             # x = torch.tensor(self.data[idx: idx + self.seq_len])
#             # y = torch.tensor(self.data[idx + self.seq_len])
#             x = self.data[idx: idx + self.seq_len]
#             y = self.data[idx + self.seq_len: idx + self.seq_len * 2]
#             # y = int(self.data[idx + self.seq_len] > self.data[idx + self.seq_len - 1])
#             # y_value = abs(self.data[idx + self.seq_len] - self.data[idx + self.seq_len - 1])
#             # y = int(self.data[idx + self.opt.group_len - 1] > self.data[idx + self.seq_len - 1])
#             # y_value = abs(self.data[idx + self.opt.group_len - 1] - self.data[idx + self.seq_len - 1])
#         else:
#             if idx >= self.real_len:
#                 # idx = (idx + np.random.ra ndint(0,self.real_len)) % self.real_len
#                 idx = idx % self.real_len
                
#             # x = self.data[idx * (self.seq_len + 1): idx * (self.seq_len + 1) + self.seq_len]
#             # y = int(self.data[(self.seq_len + 1) * idx + self.seq_len] > self.data[(self.seq_len + 1) * idx + self.seq_len - 1])
#             # y_value = abs(self.data[(self.seq_len + 1) * idx + self.seq_len] - self.data[(self.seq_len + 1) * idx + self.seq_len - 1])
#             x = self.data[idx * self.group_len: idx * self.group_len + self.seq_len]
#             y = self.data[idx * self.group_len + self.seq_len: idx * self.group_len + 2 * self.seq_len]
#             # y = int(self.data[self.group_len * idx + self.group_len - 1] > self.data[self.group_len * idx + self.seq_len - 1])
#             # y_value = abs(self.data[self.group_len * idx + self.group_len - 1] - self.data[self.group_len * idx + self.seq_len - 1])

#         return x, y, idx, self.domain_idx # , y_value


# class CovidTestDataset(Dataset):
#     def __init__(self, data, all_mean, all_std, domain_idx, opt):
#         self.data = (data - all_mean) / all_std
#         self.domain_idx = domain_idx
#         self.seq_len = opt.seq_len
#         self.opt = opt
#         self.group_len = opt.group_len

#         # Assume that we have 244 days data, so doesn't handle many speciall cases!
#         # self.real_len = int((self.data.shape[0] // self.seq_len) / 2)
#         # the last test set will have 13 datapoints. be careful!
#         # self.real_len = self.data.shape[0] // (self.seq_len + 1)
#         self.real_len = self.data.shape[0] // opt.group_len

#     def __len__(self):
#         return self.real_len

#     def __getitem__(self, idx):
#         # x = self.data[2 * self.seq_len * idx : 2 * self.seq_len * idx + self.seq_len]
#         # if idx == self.real_len - 1:
#         #     # the last test set will have 13 datapoints. be careful!
#         #     y = self.data[2 * self.seq_len * idx + self.seq_len:]
#         # else:
#         # for convenience, we only use 7 data for the last datapoints!
#         # y = self.data[2 * self.seq_len * idx +  self.seq_len : 2 * self.seq_len * idx + 2 * self.seq_len]
#         # x = self.data[idx * (self.seq_len + 1): idx * (self.seq_len + 1) + self.seq_len]
#         # y = int(self.data[(self.seq_len + 1) * idx + self.seq_len] > self.data[(self.seq_len + 1) * idx + self.seq_len - 1])
#         # y_value = abs(self.data[(self.seq_len + 1) * idx + self.seq_len] - self.data[(self.seq_len + 1) * idx + self.seq_len - 1])
#         x = self.data[idx * self.group_len: idx * self.group_len + self.seq_len]
#         y = self.data[idx * self.group_len + self.seq_len: idx * self.group_len + 2 * self.seq_len]
#         # y = int(self.data[self.group_len * idx + self.group_len - 1] > self.data[self.group_len * idx + self.seq_len - 1])
#         # y_value = abs(self.data[self.group_len * idx + self.group_len - 1] - self.data[self.group_len * idx + self.seq_len - 1])

#         return x, y, idx, self.domain_idx # , y_value