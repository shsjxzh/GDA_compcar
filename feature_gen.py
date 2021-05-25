import os
import pandas as pd 
from easydict import EasyDict
import matplotlib.pyplot as plt
import torch
import numpy as np
import random
import pickle
import torchvision.transforms as transforms
from collections import OrderedDict
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import  has_file_allowed_extension, is_image_file, IMG_EXTENSIONS, pil_loader, accimage_loader,default_loader

def write_pickle(data, name):
    with open(name,'wb') as f:
        # the default protocol level is 4
        pickle.dump(data, f)

class AlexNet_BVLC(nn.Module):
    def __init__(self, dropout=False):
        super(AlexNet_BVLC, self).__init__()
        self.features = nn.Sequential(OrderedDict([
         ("conv1", nn.Conv2d(3, 96, kernel_size=11, stride=4)),
         ("relu1", nn.ReLU(inplace=True)),
         ("pool1", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
         ("norm1", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
         ("conv2", nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)),
         ("relu2", nn.ReLU(inplace=True)),
         ("pool2", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
         ("norm2", nn.LocalResponseNorm(5, 1.e-4, 0.75)),
         ("conv3", nn.Conv2d(256, 384, kernel_size=3, padding=1)),
         ("relu3", nn.ReLU(inplace=True)),
         ("conv4", nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)),
         ("relu4", nn.ReLU(inplace=True)),
         ("conv5", nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)),
         ("relu5", nn.ReLU(inplace=True)),
         ("pool5", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),
        ]))

        self.classifier = nn.Sequential(OrderedDict([
         ("fc6", nn.Linear(256 * 6 * 6, 4096)),
         ("relu6", nn.ReLU(inplace=True)),
        #  ("drop6", nn.Dropout() if dropout else Identity(sub=0.5)),
         ("fc7", nn.Linear(4096, 4096)),
        #  ("relu7", nn.ReLU(inplace=True)),
        #  ("drop7", nn.Dropout() if dropout else Identity(sub=0.5)),
         ("fc8", nn.Linear(4096, 1000))
        ]))

        # self.final=nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier._modules['fc6'](x)
        x = self.classifier._modules['relu6'](x)
        # x = self.classifier._modules['drop6'](x)
        # x = self.classifier._modules['fc7'](x)
        # x = self.classifier._modules['relu7'](x)
        # x = self.classifier._modules['drop7'](x)
        # x = self.final(x)
        return x


    # def fix(self, alpha):
    #     for m in self.modules():
    #         if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
    #             m.momentum=alpha

    # def to_classes(self, num_classes):
    #     num_ftrs = self.final.in_features
    #     self.final = nn.Linear(num_ftrs, num_classes)
    #     self.final.weight.data.normal_(0,0.001)


def get_graph_net(url=None):
    model=AlexNet_BVLC()
    state = model.state_dict()
    state.update(torch.load(url))
    model.load_state_dict(state)
    return model


class featureDataset(Dataset):
    def __init__(self, data_lists, domain_idx, transform):
        self.domain_idx = domain_idx
        # self.sudo_len = sudo_len
        # self.file_name = opt.data_src + opt.domain_name[domain_idx]
        self.data_list = data_lists[domain_idx] # pd.read_csv(self.file_name, header=None)
        self.real_len = len(self.data_list)
        self.transform = transform

    def __len__(self):
        # do not consider sudo len
        return self.real_len


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

model = get_graph_net(url="pretrained/alexnet_caffe.pth.tar")
model.to("cuda")


# from dataset_utils.compcar_dataset import CompcarsDataLoader # SeqCovidDataLoader
# dataloader = CompcarsDataLoader(my_transform, opt) # SeqCovidDataLoader(opt)
DATES=['2009','2010','2011','2012','2013','2014']
VIEWS=['1','2','3','4','5']
domain_name = ["{}-{}".format(i,j) for i in DATES for j in VIEWS]
data_src = "data/GDA_data/"

data_lists = []
for i in range(30):
    file_name = data_src + domain_name[i]
    data_list = pd.read_csv(file_name, header=None)
    data_lists.append(data_list)


# from configs.config_compcar import opt
std=[0.229, 0.224, 0.225]
my_transform = transforms.Compose([
            transforms.Resize((224,224)),
                      transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], std)
         ])


# 1 epoch
all_encode = []
all_domain = []
all_label = []
# for i in range(30):
for i in range(30):
	# TODO: try 30 domains each
    my_dataset = featureDataset(data_lists, i, my_transform)
    dataloader = DataLoader(my_dataset, 
                    batch_size=128, 
                    shuffle=False,
                    num_workers=2,
                    # pin_memory=True,
                ) 
    for data in dataloader:
        with torch.no_grad():
            x, y, domain_idx = data 
            encode = model(x.to("cuda"))
            encode = encode.cpu().numpy()
            # print(encode.shape)
            all_encode.append(encode)
            all_domain.append(domain_idx)
            all_label.append(y)

my_data = dict()
my_data['data'] = np.concatenate(all_encode, axis=0)
print(my_data['data'].shape)
my_data['label'] = np.concatenate(all_label, axis=0)
my_data['domain'] = np.concatenate(all_domain, axis=0)

write_pickle(my_data, "data/GDA_data/feature.pkl")
