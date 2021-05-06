import os
from easydict import EasyDict
import matplotlib.pyplot as plt
import torch
import numpy as np
import random
import pickle
import torchvision.transforms as transforms
import torch.nn as nn

# import the experiment setting
# from configs.config_cluster import opt
# from configs.config_quarter import opt
# from configs.config_half_circle import opt
# from configs.config_full_circle import opt
# from configs.config_random_15 import opt
# from configs.config_random_30 import opt
# from configs.config_random_4 import opt
# from configs.config_quarter_normalize import opt
# from configs.config_half_circle_mean_equal import opt
# from configs.config_full_circle_mean_equal import opt
from configs.config_compcar import opt
# from configs.config_random_30_new import opt

# actually the config doesn't change much
# from configs.config_random_60_new import opt

np.random.seed(opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)

if opt.model == "DANN":
    from model.model import DANN as Model
elif opt.model == "GDA":
    from model.model import GDA as Model
elif opt.model == "CDANN":
    from model.model import CDANN as Model
    opt.cond_disc = True
elif opt.model == "ADDA":
    from model.model import ADDA as Model
elif opt.model == "MDD":
    from model.model import MDD as Model 
model = Model(opt).to(opt.device) # .double()

# data_source = opt.dataset

# TODO: deal with dataset that has different size!!!
# data_source = 'data/data.pickle'
# data_source = 'data/quarter-circle.pkl'
# data_source = 'data/half-circle.pkl'
# data_source = 'data/toy_d60_full.pkl'
# data_source = 'data/toy_d15_random_new.pkl'
# data_source = 'data/toy_d15_random_use.pkl'
# data_source = 'data/toy_d15_random_new_margin_1.pkl'
# data_source = 'data/toy_d30_random.pkl'

from dataset_utils.compcar_dataset import CompcarsDataLoader # SeqCovidDataLoader

std=[0.229, 0.224, 0.225]
my_transform = transforms.Compose([
            transforms.Resize((224,224)),
                      transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], std)
         ])

dataloader = CompcarsDataLoader(my_transform, opt) # SeqCovidDataLoader(opt)

# train
for epoch in range(opt.num_epoch):
    model.learn(epoch, dataloader)
    if (epoch + 1) % opt.save_interval == 0 or (epoch + 1) == opt.num_epoch:
        model.save()
        # model.visualize_D()
        # model.visualize_F()
        # model.visualize_E()
    if (epoch + 1) % opt.test_interval == 0 or (epoch + 1) == opt.num_epoch:    
        model.test(epoch, dataloader)
    # if (epoch + 1) % 1 == 0 or (epoch + 1) == opt.num_epoch:    
    #     model.test(epoch, dataloader)
    