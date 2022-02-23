import torch
import itertools
import pickle 

# from configs.opts import *
# from src.train import *
# from src.test import *
# from models.networks import get_network
import random

import copy
import numpy as np

def read_pickle(name):
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data

def write_pickle(data, name):
    with open(name,'wb') as f:
        pickle.dump(data, f)

def safe_print(x):
	try:
		print(x)
	except:
		return

# INSTANTIATE TRAINING
DATES=['2009','2010','2011','2012','2013','2014']
VIEWS=['1','2','3','4','5']

CLASSES = 4

DOMAINS = [DATES, VIEWS]
NUM_META = 2
BANDWIDTH=0.1

NUM_DOMS=1

def domain_converter(meta):
	year, viewpoint = meta
	year = int(year)-int(DATES[0])
	viewpoint = int(viewpoint)-int(VIEWS[0])
	return viewpoint*len(DATES)+year

def compute_edge(x,dt,idx, self_connection = 1.):
    edge_w=torch.exp(-torch.pow(torch.norm(x.view(1,-1)-dt,dim=1),2)/(2.*BANDWIDTH))
    edge_w[idx]=edge_w[idx]*self_connection
    return edge_w/edge_w.sum()


def get_meta_vector(meta):
	year, viewpoint = meta
	return torch.FloatTensor([float(year)-float(DATES[0]),float(viewpoint)-float(VIEWS[0])])


for i in DOMAINS:
	NUM_DOMS*=len(i)

meta_vectors = torch.FloatTensor(NUM_DOMS,NUM_META).fill_(0)
edge_vals=torch.FloatTensor(NUM_DOMS,NUM_DOMS).fill_(0)
edge_vals_no_self=torch.FloatTensor(NUM_DOMS,NUM_DOMS).fill_(0)
full_list=[]

my_index_to_ada = []
for meta in itertools.product(*DOMAINS):
    print(meta, end=' ')
    print(domain_converter(meta))
    my_index_to_ada.append(domain_converter(meta))
    full_list.append(meta)
    meta_vectors[domain_converter(meta)]=get_meta_vector(meta)

for i,vector in enumerate(meta_vectors):
    edge_vals[i,:]=compute_edge(vector,meta_vectors,i,1.)
    edge_vals_no_self[i,:]=compute_edge(vector,meta_vectors,i,0.)

print(edge_vals.shape)
print(edge_vals_no_self.shape)

np_edge_vals = edge_vals.numpy()
# print(np_edge_vals)
A = np.zeros((30, 30))
for i in range(30):
    for j in range(i + 1, 30):
        A[i][j] = np_edge_vals[my_index_to_ada[i]][my_index_to_ada[j]]
        A[j][i] = A[i][j]

print(A)
A[A < 1e-5] = 0
A[A >= 1e-5] = 1
print(A)
write_pickle(A, 'A_1e-5.pkl')

# print(A)
# for i range(6):
#     date = DATES[i]
#     for j in range(5):
#         view = VIEWS[j]
#         for k in range(30):
#             A[i * 6 + j][k] = edge_vals
