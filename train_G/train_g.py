import os
from easydict import EasyDict
import matplotlib.pyplot as plt
import torch
import numpy as np
import random
import pickle

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
import random
from visdom import Visdom

from config import opt

from GNet import GNet

# ======================================================================================================================
def to_np(x):
    return x.detach().cpu().numpy()


def to_tensor(x, device="cuda"):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).to(device)
    else:
        x = x.to(device)
    return x


def flat(x):
    n, m = x.shape[:2]
    return x.reshape(n * m, *x.shape[2:])


def write_pickle(data, name):
    with open(name, 'wb') as f:
        pickle.dump(data, f)

# ======================================================================================================================

class train_G():
    def __init__(self, opt):
        self.opt = opt
        self.device = opt.device
        self.num_domain = opt.num_domain
        self.train_log = self.opt.outf + "/loss.log"
        if not os.path.exists(self.opt.outf):
            os.mkdir(self.opt.outf)
        with open(self.train_log, 'w') as f:
            f.write("log start!\n")

        self.all_loss_G = 0

        self.epoch = 0
        # self.__set_num_domain__(opt.num_domain)

        self.netG = GNet(opt).to(opt.device)
        self.nets = [self.netG]
        
        self.__init_weight__()

        self.optimizer_G = optim.Adam(self.netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))

        self.lr_scheduler_G = lr_scheduler.ExponentialLR(optimizer=self.optimizer_G, gamma=0.5 ** (1 / 100))
        self.lr_schedulers = [self.lr_scheduler_G]

    def __init_visdom__(self):
        self.env = Visdom()
        self.pane_G = self.env.line(
            X=np.array([self.epoch]),
            Y=np.array([self.all_loss_G]),
            opts=dict(title='loss G on epochs')
        )

    def __vis_loss__(self):
        self.env.line(
            X=np.array([self.epoch]),
            Y=np.array([self.all_loss_G]),
            win=self.pane_G,
            update='append'
        )

    def __init_weight__(self):
        for net in self.nets:
            for m in net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                    nn.init.constant_(m.bias, val=0)

    def __log_write__(self, loss_msg):
        print(loss_msg)
        with open(self.train_log, 'a') as f:
            f.write(loss_msg + "\n")

    def save(self):
        if not os.path.exists(self.opt.outf):
            os.mkdir(self.opt.outf)
        for net in self.nets:
            torch.save(net.state_dict(), self.opt.outf + "/GDA_" + net.__class__.__name__)
    
    def load(self):
        for net in self.nets:
            net.load_state_dict(torch.load(self.opt.loadf + "/GDA_" + net.__class__.__name__))

    def learn(self, epoch):
        self.epoch = epoch

        self.__set_input__()
        self.__forward__()

        # # optimization
        # loss G, will be terminated after half training
        # if epoch < self.opt.num_epoch * 0.67:
        loss_G = self.__optimize_G__()
        self.all_loss_G = loss_G

        # temporary use it
        if (epoch + 1) % self.opt.show_interval == 0:
            # print("loss D: {:.4f}, loss E gan: {:.4f}, loss E pre: {:.4f}, loss G: {:.4f}".format(all_loss_D, all_loss_E_gan, all_loss_E_pred, all_loss_G))
            self.__log_write__("epoch {}, loss G: {:.4f}".format(
                self.epoch,
                self.all_loss_G)
            )

        if epoch == 0:
            self.__init_visdom__()
        else:
            self.__vis_loss__()


        # learning rate decay
        for lr_scheduler in self.lr_schedulers:
            lr_scheduler.step()
    
    def __set_input__(self):
        # one_hot for vertex
        self.t_seq = torch.nn.functional.one_hot(
            torch.arange(self.num_domain)).to(self.device)
        # self.t_seq = torch.cat(t_seq, 0).reshape(self.num_domain, self.batch_size, -1).to(self.device)

    def __forward__(self):
        for net in self.nets:
            net.train()

        self.z_seq = self.netG(self.t_seq)

    
    def __rand_walk__(self, vis, left_nodes):
        chain_node = []
        node_num = 0
        # choose node
        node_index = np.where(vis == 0)[0]
        # print(node_index)
        st = np.random.choice(node_index)
        # print(st)
        vis[st] = 1
        chain_node.append(st)
        left_nodes -= 1
        node_num += 1
        
        cur_node = st
        while left_nodes > 0:
            nx_node = -1

            node_to_choose = np.where(vis == 0)[0]
            num = node_to_choose.shape[0]
            node_to_choose = np.random.choice(node_to_choose, num, replace=False)

            for i in node_to_choose:
                if cur_node != i:
                    # have an edge and doesn't visit
                    if self.opt.A[cur_node][i] and not vis[i]:
                        nx_node = i
                        vis[nx_node] = 1
                        chain_node.append(nx_node)
                        left_nodes -= 1
                        node_num += 1
                        break
            if nx_node >= 0:
                cur_node = nx_node
            else:
                break
        # print("===chain===")
        # print(chain_node)
        # print(node_num)
        return chain_node, node_num

    def __sub_graph__(self):
        # play
        # if np.random.randint(0,2) == 0:
        #     return np.random.choice(self.num_domain, size=self.opt.sample_v, replace=False)
        
        # subsample a chain (or multiple chains in graph)
        # Todo: 需要验证指针传递还是字符串传递！！
        left_nodes = self.opt.sample_v
        choosen_node = []
        vis = np.zeros(self.num_domain)
        while left_nodes > 0:
            chain_node, node_num = self.__rand_walk__(vis, left_nodes) 
            # vis = np.zeros(self.num_domain)
            choosen_node.extend(chain_node)
            left_nodes -= node_num
        
        # print("==choosen==")
        # print(choosen_node)
        return choosen_node
        # return np.random.choice(self.num_domain, size=self.opt.sample_v, replace=False)

    def __optimize_G__(self):
        self.netG.train()

        self.optimizer_G.zero_grad()

        criterion = nn.BCEWithLogitsLoss()

        sub_graph = self.__sub_graph__()
        errorG = torch.zeros((1,)).to(self.device)
        # errorG_connected = torch.zeros((1,)).to(self.device)
        # errorG_disconnected = torch.zeros((1,)).to(self.device)
        # count_connected = 0
        # count_disconnected = 0

        sample_v = self.opt.sample_v
        
        for i in range(sample_v):
            v_i = sub_graph[i]
            for j in range(i + 1, sample_v):
                v_j = sub_graph[j]
                label = torch.tensor(self.opt.A[v_i][v_j]).to(self.device)
                # # dot product for high efficiency
                # output = (self.z_seq[v_i][0] * self.z_seq[v_j][0]).sum()
                # label = torch.full((1,), self.opt.A[v_i][v_j], device=self.device)
                # dot product
                # output = (self.z_seq[v_i] * self.z_seq[v_j]).sum(1)
                output = (self.z_seq[v_i] * self.z_seq[v_j]).sum()
                errorG += criterion(output, label)

                # # new
                # if self.opt.A[v_i][v_j]: # connected
                #     errorG_connected += criterion(output, label)
                #     count_connected += 1
                # else:
                #     errorG_disconnected += criterion(output, label)
                #     count_disconnected += 1

        # errorG = 0.5 * (errorG_connected / count_connected + errorG_disconnected / count_disconnected)
        # errorG *= self.num_domain
        # errorG = errorG / (sample_v * (sample_v - 1) / 2)
        # errorG = errorG / (sample_v * (sample_v - 1) / 2)
        # make regularization
        # errorG += 0.005 * (self.z_seq * self.z_seq).sum() / self.batch_size
        # errorG += 0.005 * nn.MSELoss()(self.z_seq, torch.zeros_like(self.z_seq)) / self.batch_size

        errorG.backward()
        
        self.optimizer_G.step()
        return errorG.item()

    def test(self, epoch):
        # pass
        for net in self.nets:
            net.eval()
        
        self.__set_input__()
        self.z_seq = self.netG(self.t_seq)


        d_all = dict()
        d_all['z'] = to_np(self.z_seq)

        write_pickle(d_all, self.opt.outf + '/' + str(epoch) + '_pred.pkl')

# =========== main code ==============
model = train_G(opt)
for epoch in range(opt.num_epoch):
    model.learn(epoch)
    if (epoch + 1) % 100 == 0 or (epoch + 1) == opt.num_epoch:
        model.save()
    if (epoch + 1) % 50 == 0 or (epoch + 1) == opt.num_epoch:    
        model.test(epoch)