import torch
import numpy as np
# G = torch.FloatTensor(
#     [
#         [1, 0],
#         [1, 1],
#         [0, 1],
#     ]
# )

# x = torch.FloatTensor(
#     [
#         [1, 0, 0],
#         [1, 0, 0],
#         [1, 0, 0],
#         [1, 0, 0],
#         [1, 0, 0],
#     ]
# )

# ans = torch.matmul(x, G)
# print(ans)

import numpy as np
# a = np.eye(3, k=1) + np.eye(3, k=-1)
# print(a)

# u = 4
# a = np.eye(u, u - 1) + np.eye(u, u -1, k=-1)
# print(a)

# v = np.random.choice(15, size=10, replace=False)
# v = np.random.choice(15)
# print(v)

# print(5 % 3)

# a = torch.zeros((1,))
# print(a)

# for i in range(5):
#     for j in range(i+1, 5):
#         print("i,j:{} {}".format(i,j))
# num_domain = 6
# a = np.eye(num_domain, k=1) + np.eye(num_domain, k=-1)
# a[0, num_domain - 1] = 1
# a[num_domain - 1, 0] =  1
# print(a)

# class basic_edge_sampler():
#     # this is a basic sampler that will use several nodes' subgraph as the sampled edge
#     def __init__(self, sample_v, num_v_all):
#         self.sample_v = sample_v
#         self.num_v_all = num_v_all

#     def __iter__(self):
#         self.i = 0
#         self.j = 0
#         self.sub_graph = self.__sub_graph__()
#         return self

#     def __next__(self):
#         self.j += 1
#         if self.j >= self.sample_v:
#             self.i += 1
#             if self.i == self.sample_v - 1:
#                 raise StopIteration
#             else:
#                 self.j = self.i + 1
        
#         return self.i, self.j

#     def __sub_graph__(self):
#         return np.random.choice(self.num_v_all, size=self.sample_v, replace=False)

# a = basic_edge_sampler(sample_v=3, num_v_all=5)
# myiter = iter(a)

# for x in a:
#     print(x)
import torch
import math
import numpy as np
from visdom import Visdom
import time

x,y=0,0
env2 = Visdom()
pane1= env2.line(
    X=np.array([x]),
    Y=np.array([y]),
    opts=dict(title='dynamic data'))

for i in range(10):
    time.sleep(1) #每隔一秒钟打印一次数据
    x+=i
    y=(y+i)*1.5
    print(x,y)
    env2.line(
        X=np.array([x]),
        Y=np.array([y]),
        win=pane1,#win参数确认使用哪一个pane
        update='append') #我们做的动作是追加，除了追加意外还有其他方式，这里我们不做介绍了

