from easydict import EasyDict
import numpy as np
import pandas as pd
import networkx as nx
# set experiment configs
import pickle
def read_pickle(name):
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data

opt = EasyDict()

# data source
opt.data_src = "data/GDA_data/"
opt.data_path = opt.data_src + "feature.pkl"
# opt.data_src = "feature.pkl"

# DATES=['2009','2010','2011','2012','2013','2014']
# VIEWS=['1','2','3','4','5']
# opt.domain_name = ["{}-{}".format(i,j) for i in DATES for j in VIEWS]

opt.model = "GDA"
# opt.model = "DANN"
# opt.model = "CDANN"
# opt.model = 'ADDA'
# opt.model = 'MDD'

if opt.model == 'CDANN':
    opt.cond_disc = True
else:
    opt.cond_disc = False


print("model: {}".format(opt.model))

opt.use_visdom = True # False # True # 
opt.visdom_port = 3000

opt.sample_neighbour = False

opt.seed = 333 # 223

# opt.use_pretrain_f = True
# # the toy dataset will temporarily not use pretrained G
# if opt.use_pretrain_f:
#     opt.feature_path = 'pretrained/alexnet_caffe.pth.tar'

opt.use_g_encode = True # False
if opt.use_g_encode:
    opt.g_encode = read_pickle("derive_g_encode/g_encode.pkl")

# dataset and data loading
opt.shuffle=True

opt.src_domain = [0,1,5] # list(range(0,12))
opt.tgt_domain = [2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]# list(range(12,30))

opt.src_dmn_num = len(opt.src_domain)
opt.tgt_dmn_num = len(opt.tgt_domain)

opt.all_domain = opt.src_domain + opt.tgt_domain
opt.num_domain = len(opt.all_domain)

# often used training setting
# G and D balance
opt.lambda_gan = 0.5 # 0.2

# for MDD
opt.lambda_src = 0.5
opt.lambda_tgt = 0.5

# opt.lambda_reg = 20
opt.batch_size = 16
# opt.test_batch_size = 100
opt.num_epoch = 6
opt.lr_e = 1 * 1e-4 # 1e-4
opt.lr_d = 1 * 1e-4
opt.lr_g = 1e-3
opt.wgan = False
opt.outf = "dump"
opt.test_on_all_dmn = True # whethor test on all the domains or just target domain
# opt.group_len = 12

opt.save_interval = 10 # not use !!
opt.test_interval = 2 # 20 # 5
opt.iter_interval = 10

# modules setting
opt.no_bn = True
opt.gamma = 100
opt.beta1 = 0.9
opt.weight_decay = 5e-4

opt.num_input = 4096    # the x data dimension
opt.nh = 4096        # TODO: the hidden states for many modules, be careful
opt.nc = 4           # num classes, for compcar, it is 4
opt.nd_out = 2       # the output dimension of D
opt.nv_embed = 2     # the vertex embedding dimension
opt.p=0.2



# sampling number
opt.sample_v =  20 # 27 # 27 # 4 # 14 # 27 # 14     # the sample number for d's training
opt.sample_v_g =  30 # 48 # 48 # 4 # 21 # 48 # 21   # the sample number for g's training

opt.device = "cuda" # "cuda"

# opt.A = read_pickle(opt.data_src + "new_A.pkl")
# opt.A = read_pickle(opt.data_src + "A_1e-6.pkl")
opt.A = read_pickle(opt.data_src + "A.pkl")
# for A.pkl only!!!!
opt.A[opt.A >= 1e-5] = 1
opt.A[opt.A < 1e-5] = 0
print(opt.A)
