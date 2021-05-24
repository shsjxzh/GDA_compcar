import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from model.modules import *
import pickle
import random
from visdom import Visdom
import time

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

# the base model
class BaseModel(nn.Module):
    def __init__(self, opt):
        super(BaseModel, self).__init__()
        # set output format
        np.set_printoptions(suppress=True, precision=6)

        self.opt = opt
        self.device = opt.device
        self.batch_size = opt.batch_size
        # visualizaiton
        self.use_visdom = opt.use_visdom
        self.use_g_encode = opt.use_g_encode
        if opt.use_visdom:
            self.env = Visdom(port=opt.visdom_port)
            self.test_pane = dict()

        # self.seq_len = opt.seq_len
        # self.group_len = opt.group_len
        # TODO: check whether num domain is needed
        self.num_domain = opt.num_domain
        if self.opt.test_on_all_dmn:
            self.test_dmn_num = self.num_domain
        else:
            self.test_dmn_num = self.opt.tgt_dmn_num
        
        self.train_log = self.opt.outf + "/loss.log"
        self.model_path = opt.outf + '/model.pth'
        self.out_pic_f = opt.outf + '/plt_pic'
        if not os.path.exists(self.opt.outf):
            os.mkdir(self.opt.outf)
        if not os.path.exists(self.out_pic_f):
            os.mkdir(self.out_pic_f)
        with open(self.train_log, 'w') as f:
            f.write("log start!\n")

        # TODO: need more code here
        # 这里需要用self loop中的source domain定义方式
        mask_list = np.zeros(opt.num_domain)
        mask_list[opt.src_domain] = 1
        self.domain_mask = torch.IntTensor(mask_list).to(opt.device)  # not sure if device is needed

        # used for plot
        self.all_count = 0

    
    def learn(self, epoch, dataloader):
        self.train()
        
        self.epoch = epoch
        loss_values = {
            loss: 0 for loss in self.loss_names
        }


        # start = time.time()

        count = 0
        for data in dataloader.get_train_data():
            # print(data)
            count += 1
            self.all_count += 1
            # print(count)
            self.__set_input__(data)
            # print("finish set up")
            self.__train_forward__()
            # print("finish train forward")
            new_loss_values = self.__optimize__()
            # print("finish optimize")

            # for the loss visualization
            for key, loss in new_loss_values.items():
                loss_values[key] += loss

            # if self.use_visdom and (self.epoch + 1) % 10 == 0:
            #     self.__vis_loss__(loss)

            # # correct form:
            # if self.use_visdom and count % self.opt.iter_interval == 0:
            #     self.__vis_count_loss__(new_loss_values)
            
            # # break
            # if count >= 10:
            #     break
            
        
        # end = time.time()
        # print("time: {}".format(end - start))

        for key, _ in new_loss_values.items():
            loss_values[key] /= count

        if self.use_visdom:
            self.__vis_loss__(loss_values)

        # if (self.epoch + 1) % 10 == 0:
            # print("epoch {}: {}".format(self.epoch,loss_values))
        print("epoch {}: {}".format(self.epoch,loss_values))

        # learning rate decay
        # tmp not use !!
        for lr_scheduler in self.lr_schedulers:
            lr_scheduler.step()
        
        # print("finish train")


    def test(self, epoch, dataloader):
        # pass
        self.eval()
        self.epoch = epoch
        
        # now is the embedding printing version
        acc_curve = []
        # do not save any x
        # l_x = []
        # l_y = []
        l_domain = []
        # l_prob = []
        l_label = []
        l_encode = []
        # l_decode = []
        # l_z_seq = []
        # big change on l_z_seq !!!!
        # z_seq = 
        z_seq = 0

        for data in dataloader.get_test_data():
            self.__set_input__(data)
            # forward
            with torch.no_grad():
                # z_seq = self.netG(self.t_seq)
                # e_seq = self.netE(self.x_seq ,z_seq)
                # f_seq = self.netF(e_seq)
                # g_seq = torch.argmax(f_seq.detach(), dim=2)  # class of the prediction
                # # 使用test forward的改进
                # d_seq = self.netD(e_seq)
                self.__test_forward__()

            # z_seq = to_np(self.z_seq)
            # print(z_seq)
            if isinstance(z_seq,int):
                z_seq = to_np(self.z_seq)

            # print(z_seq.shape)

            acc_curve.append(self.g_seq.eq(self.y_seq).to(torch.float).mean(-1, keepdim=True))

            # if self.opt.normalize_domain:
            #     # still working on normalize domain
            #     # pass
            #     x_np = to_np(self.x_seq)
            #     for i in range(len(x_np)):
            #         x_np[i] = x_np[i] * self.data_s[i] + self.data_m[i]
            #     l_x.append(x_np)
            # else:
            #     l_x.append(to_np(self.x_seq))

            # l_y.append(to_np(y_seq))
            # l_z_seq.append(to_np(z_seq))
            l_domain.append(to_np(self.domain_seq))
            # l_prob.append(to_np(self.f_seq))
            l_encode.append(to_np(self.e_seq))
            # l_decode.append(to_np(d_seq))
            l_label.append(to_np(self.g_seq))

            # break

        # x_all = np.concatenate(l_x, axis=1) not 
        e_all = np.concatenate(l_encode, axis=1)
        # decode_all = np.concatenate(l_decode, axis=1)
        # y_all = np.concatenate(l_y, axis=1)
        domain_all = np.concatenate(l_domain, axis=1)
        # prob_all = np.concatenate(l_prob, axis=1)
        label_all = np.concatenate(l_label, axis=1)

        # print(np.asarray(l_z_seq).shape)
        # z_seq = to_np(self.z_seq)

        # print(z_seq.shape)
        z_seq_all = z_seq[0:self.batch_size * self.test_dmn_num:self.batch_size,:]
        # print(z_seq_all.shape)
        # print(z_seq_all.shape)
        # z_seq_all = np.concatenate(l_z_seq, axis=1)

        # print(z_seq_all.shape)
        # print(label_all.shape)
        # print(x_all.shape)

        d_all = dict()

        # d_all['data'] = flat(x_all)
        # d_all['gt'] = flat(y_all)
        d_all['domain'] = flat(domain_all)
        # d_all['prob'] = flat(prob_all)
        d_all['label'] = flat(label_all)
        d_all['encodeing'] = flat(e_all)
        # d_all['decodeing'] = flat(decode_all)
        d_all['z'] = z_seq_all
        # clean the g bias term
        # d_all['g_bias'] = to_np(self.netG.bias)
        # d_all['g_weight'] = to_np(self.netG.weight)

        acc = to_np(torch.cat(acc_curve, 1).mean(-1))
        # test_acc = (acc.sum() - acc[self.opt.src_domain].sum()) / (self.opt.tgt_dmn_num) * 100
        test_acc = acc[self.opt.tgt_domain].sum() / (self.opt.tgt_dmn_num) * 100
        acc_msg = '[Test][{}] Accuracy: total average {:.1f}, test average {:.1f}, in each domain {}'.format(epoch, acc.mean() * 100, test_acc, np.around(acc * 100, decimals=1))
        # print(acc_msg)
        self.__log_write__(acc_msg)
        if self.use_visdom:
            self.__vis_test_error__(test_acc, 'test acc')

        d_all['acc_msg'] = acc_msg

        write_pickle(d_all, self.opt.outf + '/' + str(epoch) + '_pred.pkl')

        print("finish test")

    def __vis_test_error__(self, loss, title):
        if self.epoch == self.opt.test_interval - 1:
            # if self.test_pane is None:
            #     self.test_pane = dict()
            self.test_pane[title] = self.env.line(
                                X=np.array([self.epoch]),
                                Y=np.array([loss]),
                                opts=dict(title=title)
                            )
        else:
            self.env.line(
                X=np.array([self.epoch]),
                Y=np.array([loss]),
                win=self.test_pane[title],
                update='append'
            )

    def save(self):
        torch.save(self.state_dict(), self.model_path)

    def __set_input__(self, data, train=True):
        """
        :param
            x_seq: Number of domain x Batch size x length
            y_seq: Number of domain x Batch size x Predict Data dim
            (testing: Number of domain x Batch size x test len x Predict Data dim)
            one_hot_seq: Number of domain x Batch size x Number of vertices (domains)
            domain_seq: Number of domain x Batch size x domain dim (1)
            idx_seq: Number of domain x Batch size x 1 (the order in the whole dataset)
            y_value_seq: Number of domain x Batch size x Predict Data dim
        """
        # a tmp operation to add dim for data
        if train:
            # the domain seq is in d3!!
            # x_seq, y_seq, domain_seq = [d[0][None, :, :, :, :] for d in data], [d[1][None, :] for d in data], [d[2][None, :] for d in data] # , [d[3][None, :] for d in data]

            x_seq, y_seq, domain_seq = [d[0][None, :, :] for d in data], [d[1][None, :] for d in data], [d[2][None, :] for d in data]

            self.x_seq = torch.cat(x_seq, 0).to(self.device) # .double()
            self.y_seq = torch.cat(y_seq, 0).to(self.device) # .unsqueeze(-1) # .double())
            # self.idx_seq = np.concatenate(idx_seq, 0)
            self.domain_seq = torch.cat(domain_seq, 0).to(self.device)
            self.tmp_batch_size = self.x_seq.shape[1]
            one_hot_seq = [torch.nn.functional.one_hot(d[2], self.num_domain) for d in data]
            self.one_hot_seq = torch.cat(one_hot_seq, 0).reshape(self.num_domain, self.tmp_batch_size, -1).to(self.device)

        else:
            # x_seq, y_seq, domain_seq = [d[0][None, :, :, :, :] for d in data], [d[1][None, :] for d in data], [d[2][None, :] for d in data] # , [d[3][None, :] for d in data]
            x_seq, y_seq, domain_seq = [d[0][None, :, :] for d in data], [d[1][None, :] for d in data], [d[2][None, :] for d in data]

            self.x_seq = torch.cat(x_seq, 0).to(self.device) # .double()
            self.y_seq = torch.cat(y_seq, 0).to(self.device) # .unsqueeze(-1) # .double()
            # self.idx_seq = np.concatenate(idx_seq, 0)
            self.domain_seq = torch.cat(domain_seq, 0).to(self.device)
            self.tmp_batch_size = self.x_seq.shape[1]

            # this method is ugly but keep the same with the training
            # print(self.num_domain)
            one_hot_seq = [torch.nn.functional.one_hot(d[2], self.num_domain) for d in data]
            self.one_hot_seq = torch.cat(one_hot_seq, 0).reshape(self.test_dmn_num, self.tmp_batch_size, -1).to(self.device)


    def __train_forward__(self):
        # TODO: implement the net
        self.z_seq = self.netG(self.one_hot_seq)        # be sure about the one_hot_seq meaning!
        # self.e_seq = self.netE(self.x_seq, self.z_seq)  # encoder of the data
        self.e_seq = self.netE(self.x_seq, self.z_seq)  # encoder of the data
        self.f_seq = self.netF(self.e_seq)              # regression for prediction
        # self.d_seq = self.netD(self.e_seq)

        if self.opt.lambda_gan != 0:
            self.d_seq = self.netD(self.e_seq)

            # this is the d loss, still not backward yet
            self.loss_D = self.__loss_D__(self.d_seq)

    def __test_forward__(self):
        self.z_seq = self.netG(self.one_hot_seq)        # be sure about the one_hot_seq meaning!
        self.e_seq = self.netE(self.x_seq, self.z_seq)  # encoder of the data
        self.f_seq = self.netF(self.e_seq)
        self.g_seq = torch.argmax(self.f_seq.detach(), dim=2)  # class of the prediction
        # self.f_value_seq = self.netF_value(self.e_seq)   
        # self.g_seq = torch.argmax(self.f_seq.detach(), dim=2)

    def __optimize__(self):
        loss_value = dict()
        if not self.use_g_encode:
            loss_value['G'] = self.__optimize_G__()
        if self.opt.lambda_gan != 0:
            loss_value['D'] = self.__optimize_D__()
        else:
            loss_value['D'] = 0
        # print(loss_value['D'])
        loss_value['E_pred'], loss_value['E_gan'] = self.__optimize_EF__() # loss_value['E_pred_value'], 
        
        if self.opt.wgan:
            clamp_range = 2.0
            for p in self.netD.parameters():
                p.data.clamp_(-clamp_range, clamp_range)
        
        return loss_value

    def __optimize_G__(self):
        self.netG.train()
        self.netD.eval(), self.netE.eval(), self.netF.eval(), # self.netF_value.eval()

        self.optimizer_G.zero_grad()

        criterion = nn.BCEWithLogitsLoss()

        sub_graph = self.__sub_graph__(my_sample_v=self.opt.sample_v_g)
        errorG = torch.zeros((1,)).to(self.device) # .double()
        sample_v = self.opt.sample_v_g
        # train_z_seq = self.z_seq.reshape()
        
        for i in range(sample_v):
            v_i = sub_graph[i]
            for j in range(i + 1, sample_v):
                v_j = sub_graph[j]
                # label = torch.tensor(self.opt.A[v_i][v_j]).to(self.device)
                # label = torch.full((self.batch_size,), self.opt.A[v_i][v_j], device=self.device)
                label = torch.tensor(self.opt.A[v_i][v_j]).to(self.device) # , dtype=torch.double
                # dot product
                # output = self.netG.weight * (self.z_seq[v_i * self.batch_size] * self.z_seq[v_j * self.batch_size]).sum() + self.netG.bias
                output = (self.z_seq[v_i * self.tmp_batch_size] * self.z_seq[v_j * self.tmp_batch_size]).sum()
                errorG += criterion(output, label)

        errorG /= (sample_v * (sample_v - 1) / 2)

        errorG.backward(retain_graph=True)
        
        self.optimizer_G.step()
        return errorG.item()

    def __optimize_D__(self):
        self.netD.train()
        self.netG.eval(), self.netE.eval(), self.netF.eval(), # self.netF_value.eval()

        self.optimizer_D.zero_grad()

        # backward process:
        self.loss_D.backward(retain_graph=True)

        self.optimizer_D.step()
        return self.loss_D.item()

    def __optimize_EF__(self):
        self.netD.eval(), self.netG.eval()
        self.netE.train(), self.netF.train()

        # self.__set_requires_grad__(self.netD, False)
        self.optimizer_EF.zero_grad()

        if self.opt.lambda_gan != 0:
            loss_E_gan = - self.loss_D
        else:
            loss_E_gan = torch.tensor(0, dtype=torch.float, device=self.opt.device)

        y_seq_source = self.y_seq[self.domain_mask == 1]
        f_seq_source = self.f_seq[self.domain_mask == 1]

        loss_E_pred = F.nll_loss(flat(f_seq_source), flat(y_seq_source))
        
        loss_E = loss_E_gan * self.opt.lambda_gan + loss_E_pred
        loss_E.backward()

        self.optimizer_EF.step()

        return loss_E_pred.item(), loss_E_gan.item()

    def __log_write__(self, loss_msg):
        print(loss_msg)
        with open(self.train_log, 'a') as f:
            f.write(loss_msg + "\n")

    def __vis_loss__(self, loss_values):
        if self.epoch == 0:
            self.panes = {
                loss_name: 
                self.env.line(
                    X=np.array([self.epoch]),
                    Y=np.array([loss_values[loss_name]]),
                    opts=dict(title='loss for {} on epochs'.format(loss_name))
                )
                for loss_name in self.loss_names
            }
        else:
            for loss_name in self.loss_names:
                self.env.line(
                    X=np.array([self.epoch]),
                    Y=np.array([loss_values[loss_name]]),
                    win=self.panes[loss_name],
                    update='append'
                )
    
    def __vis_count_loss__(self, loss_values):
        if self.all_count == self.opt.iter_interval:
            self.count_panes = {
                loss_name: 
                self.env.line(
                    X=np.array([self.all_count]),
                    Y=np.array([loss_values[loss_name]]),
                    opts=dict(title='loss for {} on iters'.format(loss_name))
                )
                for loss_name in self.loss_names
            }
        else:
            for loss_name in self.loss_names:
                self.env.line(
                    X=np.array([self.all_count]),
                    Y=np.array([loss_values[loss_name]]),
                    win=self.count_panes[loss_name],
                    update='append'
                )


    def __init_weight__(self, net=None):
        if net is None:
            net = self
        for m in net.modules():
            if isinstance(m, nn.Linear):
                # print("init linear weight!")
                nn.init.normal_(m.weight, mean=0, std=0.01)
#                 nn.init.normal_(m.weight, mean=0, std=0.1)
#                 nn.init.xavier_normal_(m.weight, gain=10)
                nn.init.constant_(m.bias, val=0)

    # for graph random sampling:
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

    def __sub_graph__(self, my_sample_v):
        if self.opt.sample_neighbour:
            # print("sample neighbour")
            return self.__neighbour__()
            

        # play
        if np.random.randint(0,2) == 0:
            # for debugging, temporarily close the replacement
            return np.random.choice(self.num_domain, size=my_sample_v, replace=False)
        
        # subsample a chain (or multiple chains in graph)
        # Todo: 需要验证指针传递还是字符串传递！！
        # left_nodes = self.opt.sample_v
        left_nodes = my_sample_v
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
    
    def __neighbour__(self):
        # find the neighbours?
        # 单层neighbour
        choosen_node = []
        st = np.random.randint(0, self.num_domain)
        choosen_node.append(st)
        for i in range(self.num_domain):
            if self.opt.A[st][i]:
                choosen_node.append(i)
        
        # must have elements because the graph is connected
        node_new = choosen_node[1:]
        for node in node_new:
            for i in range(self.num_domain):
                if self.opt.A[node][i] and i not in choosen_node:
                    choosen_node.append(i)
        
        # print(choosen_node)

        return choosen_node




class DANN(BaseModel):
    """
    DANN Model
    """
    def __init__(self, opt):
        super(DANN, self).__init__(opt)
        self.netE = FeatureNet(opt).to(opt.device)

        # self.netE = AlexNet_BVLC_Feature(opt).to(opt.device) # GRUSeqFeatureNet(opt).to(opt.device)
        self.netF = PredNet(opt).to(opt.device) # GRUSeqPredNet(opt).to(opt.device)
        self.netG = GNet(opt).to(opt.device)
        self.netD = ClassDiscNet(opt).to(opt.device)

        self.__init_weight__()

        # if opt.use_pretrain_f:
        #     self.netE.load_state_dict(torch.load(opt.feature_path), strict=False)

        EF_parameters = list(self.netE.parameters()) + list(self.netF.parameters())
        self.optimizer_EF= optim.Adam(EF_parameters, lr=opt.lr_e, betas=(opt.beta1, 0.999))
        self.optimizer_D = optim.Adam(self.netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
        # temporary just try the 
        if not self.use_g_encode:
            self.optimizer_G =  optim.Adam(self.netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))

        self.lr_scheduler_EF = lr_scheduler.ExponentialLR(optimizer=self.optimizer_EF, gamma=0.5 ** (1/100))
        self.lr_scheduler_D = lr_scheduler.ExponentialLR(optimizer=self.optimizer_D, gamma=0.5 ** (1/100))
        if not self.use_g_encode:
            self.lr_scheduler_G = lr_scheduler.ExponentialLR(optimizer=self.optimizer_G, gamma=0.5 ** (1 / 100))
            self.lr_schedulers = [self.lr_scheduler_EF, self.lr_scheduler_D, self.lr_scheduler_G]
        else:
            self.lr_schedulers = [self.lr_scheduler_EF, self.lr_scheduler_D]
        self.loss_names = ['E_pred','E_gan', 'D', 'G']
        if self.use_g_encode:
            self.loss_names.remove('G')

        self.lambda_gan = self.opt.lambda_gan

    def __train_forward__(self):
        # TODO: implement the net
        self.z_seq = self.netG(self.one_hot_seq)        # be sure about the one_hot_seq meaning!
        # self.e_seq = self.netE(self.x_seq, self.z_seq)  # encoder of the data
        self.e_seq = self.netE(self.x_seq, self.z_seq)  # encoder of the data
        self.f_seq = self.netF(self.e_seq)

    def __optimize_D__(self):
        self.netD.train()
        self.netG.eval(), self.netE.eval(), self.netF.eval(), # self.netF_value.eval()

        self.optimizer_D.zero_grad()

        # backward process:
        # self.loss_D.backward(retain_graph=True)
        self.d_seq = self.netD(self.e_seq.detach())
        # print(self.d_seq.shape)
        self.loss_D = F.nll_loss(flat(self.d_seq), flat(self.domain_seq))
        self.loss_D.backward()

        self.optimizer_D.step()
        return self.loss_D.item()

    def __optimize_EF__(self):
        self.netD.eval(), self.netG.eval()
        self.netE.train(), self.netF.train(), # self.netF_value.train()

        self.optimizer_EF.zero_grad()
        # criterion = nn.MSELoss()
        self.d_seq = self.netD(self.e_seq)

        self.loss_E_gan = - F.nll_loss(flat(self.d_seq), flat(self.domain_seq))

        self.y_seq_source = self.y_seq[self.domain_mask == 1]
        self.f_seq_source = self.f_seq[self.domain_mask == 1]

        self.loss_E_pred = F.nll_loss(flat(self.f_seq_source), flat(self.y_seq_source))

        self.loss_E = self.loss_E_gan * self.lambda_gan + self.loss_E_pred
        self.loss_E.backward()
        self.optimizer_EF.step()

        return self.loss_E_pred.item(), self.loss_E_gan.item() # loss_E_pred_value.item(), 

    

class CDANN(BaseModel):
    """
    CDANN Model
    """
    def __init__(self, opt):
        super(CDANN, self).__init__(opt)
        self.netE = FeatureNet(opt).to(opt.device)
        # self.netE = AlexNet_BVLC_Feature(opt).to(opt.device) # GRUSeqFeatureNet(opt).to(opt.device)
        self.netF = PredNet(opt).to(opt.device) # GRUSeqPredNet(opt).to(opt.device)
        self.netG = GNet(opt).to(opt.device)
        self.netD = CondClassDiscNet(opt).to(opt.device)

        self.__init_weight__()
        # if opt.use_pretrain_f:
        #     self.netE.load_state_dict(torch.load(opt.feature_path), strict=False)

        EF_parameters = list(self.netE.parameters()) + list(self.netF.parameters())
        self.optimizer_EF= optim.Adam(EF_parameters, lr=opt.lr_e, betas=(opt.beta1, 0.999))
        self.optimizer_D = optim.Adam(self.netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
        # temporary just try the
        if not self.use_g_encode: 
            self.optimizer_G =  optim.Adam(self.netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))

        self.lr_scheduler_EF = lr_scheduler.ExponentialLR(optimizer=self.optimizer_EF, gamma=0.5 ** (1/100))
        self.lr_scheduler_D = lr_scheduler.ExponentialLR(optimizer=self.optimizer_D, gamma=0.5 ** (1/100))
        if not self.use_g_encode:
            self.lr_scheduler_G = lr_scheduler.ExponentialLR(optimizer=self.optimizer_G, gamma=0.5 ** (1 / 100))
            self.lr_schedulers = [self.lr_scheduler_EF, self.lr_scheduler_D, self.lr_scheduler_G]
        else:
            self.lr_schedulers = [self.lr_scheduler_EF, self.lr_scheduler_D]
        self.loss_names = ['E_pred','E_gan', 'D', 'G']
        if self.use_g_encode:
            self.loss_names.remove('G')

        self.lambda_gan = self.opt.lambda_gan

    def __train_forward__(self):
        # TODO: implement the net
        self.z_seq = self.netG(self.one_hot_seq)        # be sure about the one_hot_seq meaning!
        # self.e_seq = self.netE(self.x_seq, self.z_seq)  # encoder of the data
        self.e_seq = self.netE(self.x_seq, self.z_seq)  # encoder of the data
        self.f_seq = self.netF(self.e_seq)
        self.f_seq_sig = torch.sigmoid(self.f_seq.detach())

    def __optimize_D__(self):
        self.netD.train()
        self.netG.eval(), self.netE.eval(), self.netF.eval(), # self.netF_value.eval()

        self.optimizer_D.zero_grad()

        # backward process:
        # self.loss_D.backward(retain_graph=True)
        self.d_seq = self.netD(self.e_seq.detach(), self.f_seq_sig)
        # print(self.d_seq.shape)
        self.loss_D = F.nll_loss(flat(self.d_seq), flat(self.domain_seq))
        self.loss_D.backward()

        self.optimizer_D.step()
        return self.loss_D.item()

    def __optimize_EF__(self):
        self.netD.eval(), self.netG.eval()
        self.netE.train(), self.netF.train(), # self.netF_value.train()

        self.optimizer_EF.zero_grad()
        # criterion = nn.MSELoss()
        self.d_seq = self.netD(self.e_seq, self.f_seq_sig)

        self.loss_E_gan = - F.nll_loss(flat(self.d_seq), flat(self.domain_seq))

        self.y_seq_source = self.y_seq[self.domain_mask == 1]
        self.f_seq_source = self.f_seq[self.domain_mask == 1]

        self.loss_E_pred = F.nll_loss(flat(self.f_seq_source), flat(self.y_seq_source))

        self.loss_E = self.loss_E_gan * self.lambda_gan + self.loss_E_pred
        self.loss_E.backward()
        self.optimizer_EF.step()

        return self.loss_E_pred.item(), self.loss_E_gan.item() # loss_E_pred_value.item(), 


class ADDA(BaseModel):
    def __init__(self, opt):
        super(ADDA, self).__init__(opt)
        self.netE = FeatureNet(opt).to(opt.device)
        # self.netE = AlexNet_BVLC_Feature(opt).to(opt.device) # GRUSeqFeatureNet(opt).to(opt.device)
        self.netF = PredNet(opt).to(opt.device) # GRUSeqPredNet(opt).to(opt.device)
        self.netG = GNet(opt).to(opt.device)
        self.netD = DiscNet(opt).to(opt.device)
        
        self.__init_weight__()
        # if opt.use_pretrain_f:
        #     self.netE.load_state_dict(torch.load(opt.feature_path), strict=False)

        EF_parameters = list(self.netE.parameters()) + list(self.netF.parameters())
        self.optimizer_EF= optim.Adam(EF_parameters, lr=opt.lr_e, betas=(opt.beta1, 0.999))
        self.optimizer_D = optim.Adam(self.netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
        # temporary just try the 
        if not self.use_g_encode:
            self.optimizer_G =  optim.Adam(self.netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))

        self.lr_scheduler_EF = lr_scheduler.ExponentialLR(optimizer=self.optimizer_EF, gamma=0.5 ** (1/100))
        self.lr_scheduler_D = lr_scheduler.ExponentialLR(optimizer=self.optimizer_D, gamma=0.5 ** (1/100))
        if not self.use_g_encode:
            self.lr_scheduler_G = lr_scheduler.ExponentialLR(optimizer=self.optimizer_G, gamma=0.5 ** (1 / 100))
            self.lr_schedulers = [self.lr_scheduler_EF, self.lr_scheduler_D, self.lr_scheduler_G]
        else:
            self.lr_schedulers = [self.lr_scheduler_EF, self.lr_scheduler_D]
        self.loss_names = ['E_pred','E_gan', 'D', 'G']
        if self.use_g_encode:
            self.loss_names.remove('G')

    def __train_forward__(self):
        # TODO: implement the net
        self.z_seq = self.netG(self.one_hot_seq)        # be sure about the one_hot_seq meaning!
        # self.e_seq = self.netE(self.x_seq, self.z_seq)  # encoder of the data
        self.e_seq = self.netE(self.x_seq, self.z_seq)  # encoder of the data
        self.f_seq = self.netF(self.e_seq)

    def __optimize_D__(self):
        self.netD.train()
        self.netG.eval(), self.netE.eval(), self.netF.eval(), # self.netF_value.eval()

        self.optimizer_D.zero_grad()

        # backward process:
        # self.loss_D.backward(retain_graph=True)
        self.d_seq = self.netD(self.e_seq.detach())
        self.d_seq_source = self.d_seq[self.domain_mask == 1]
        self.d_seq_target = self.d_seq[self.domain_mask == 0]
        # D: discriminator loss from classifying source v.s. target
        self.loss_D = - torch.log(self.d_seq_source + 1e-10).mean() \
                      - torch.log(1 - self.d_seq_target + 1e-10).mean()
        self.loss_D.backward()

        self.optimizer_D.step()
        return self.loss_D.item()

    def __optimize_EF__(self):
        self.netD.eval(), self.netG.eval()
        self.netE.train(), self.netF.train(), # self.netF_value.train()

        self.optimizer_EF.zero_grad()
        # criterion = nn.MSELoss()
        self.d_seq = self.netD(self.e_seq)
        self.d_seq_target = self.d_seq[self.domain_mask == 0]
        self.loss_E_gan = - torch.log(self.d_seq_target + 1e-10).mean()
        # E_pred: encoder loss from prediction the label
        self.y_seq_source = self.y_seq[self.domain_mask == 1]
        self.f_seq_source = self.f_seq[self.domain_mask == 1]
        self.loss_E_pred = F.nll_loss(flat(self.f_seq_source), flat(self.y_seq_source))

        self.loss_E = self.loss_E_gan * self.opt.lambda_gan + self.loss_E_pred

        self.optimizer_EF.step()

        return self.loss_E_pred.item(), self.loss_E_gan.item() # loss_E_pred_value.item(), 


class MDD(BaseModel):
    '''
    Margin Disparity Discrepancy
    '''
    def __init__(self, opt):
        super(MDD, self).__init__(opt)
        self.netE = FeatureNet(opt).to(opt.device)
        # self.netE = AlexNet_BVLC_Feature(opt).to(opt.device) # GRUSeqFeatureNet(opt).to(opt.device)
        self.netF = PredNet(opt).to(opt.device) # GRUSeqPredNet(opt).to(opt.device)
        self.netG = GNet(opt).to(opt.device)
        self.netD = PredNet(opt).to(opt.device)

        self.__init_weight__()
        # if opt.use_pretrain_f:
        #     self.netE.load_state_dict(torch.load(opt.feature_path), strict=False)

        EF_parameters = list(self.netE.parameters()) + list(self.netF.parameters())
        self.optimizer_EF= optim.Adam(EF_parameters, lr=opt.lr_e, betas=(opt.beta1, 0.999))
        self.optimizer_D = optim.Adam(self.netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
        # temporary just try the 
        if not self.use_g_encode:
            self.optimizer_G =  optim.Adam(self.netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))

        self.lr_scheduler_EF = lr_scheduler.ExponentialLR(optimizer=self.optimizer_EF, gamma=0.5 ** (1/100))
        self.lr_scheduler_D = lr_scheduler.ExponentialLR(optimizer=self.optimizer_D, gamma=0.5 ** (1/100))
        if not self.use_g_encode:
            self.lr_scheduler_G = lr_scheduler.ExponentialLR(optimizer=self.optimizer_G, gamma=0.5 ** (1 / 100))
            self.lr_schedulers = [self.lr_scheduler_EF, self.lr_scheduler_D, self.lr_scheduler_G]
        else:
            self.lr_schedulers = [self.lr_scheduler_EF, self.lr_scheduler_D]
        self.loss_names = ['E_pred','E_adv', 'ADV_src', 'ADV_tgt', 'G']
        if self.use_g_encode:
            self.loss_names.remove('G')

        self.lambda_src = opt.lambda_src
        self.lambda_tgt = opt.lambda_tgt

    def __train_forward__(self):
        # TODO: implement the net
        self.z_seq = self.netG(self.one_hot_seq)        # be sure about the one_hot_seq meaning!
        # self.e_seq = self.netE(self.x_seq, self.z_seq)  # encoder of the data
        self.e_seq = self.netE(self.x_seq, self.z_seq)  # encoder of the data
        self.f_seq = self.netF(self.e_seq)
        self.g_seq = torch.argmax(self.f_seq.detach(), dim=2)  # class of the prediction

    def __optimize__(self):
        loss_value = dict()
        if not self.use_g_encode:
            loss_value['G'] = self.__optimize_G__()
        if self.opt.lambda_gan != 0:
            loss_value['ADV_src'], loss_value['ADV_tgt'] = self.__optimize_D__()
        else:
            loss_value['ADV_src'], loss_value['ADV_tgt'] = 0
        # print(loss_value['D'])
        loss_value['E_pred'], loss_value['E_adv'] = self.__optimize_EF__() # loss_value['E_pred_value'], 
        return loss_value

    def __optimize_D__(self):
        self.netD.train()
        self.netG.eval(), self.netE.eval(), self.netF.eval(), # self.netF_value.eval()

        self.optimizer_D.zero_grad()

        # # backward process:
        # self.loss_D.backward(retain_graph=True)
        # return self.loss_D.item()
        self.f_adv, self.f_adv_softmax = self.netD(self.e_seq.detach(), return_softmax=True)
        # agreement with netF on source domain
        self.loss_ADV_src = F.nll_loss(flat(self.f_adv[self.domain_mask == 1]),
                                       flat(self.g_seq[self.domain_mask == 1]))
        f_adv_tgt = torch.log(1 - self.f_adv_softmax[self.domain_mask == 0] + 1e-10)
        # disagreement with netF on target domain
        self.loss_ADV_tgt = F.nll_loss(flat(f_adv_tgt),
                                       flat(self.g_seq[self.domain_mask == 0]))
        # minimize the agreement on source domain while maximize the disagreement on target domain
        self.loss_D = (self.loss_ADV_src * self.lambda_src + self.loss_ADV_tgt * self.lambda_tgt) / (self.lambda_src + self.lambda_tgt)

        self.loss_D.backward()
        # self.f_adv = self.netD(self.e_seq.detach())
        # loss_ADV_src = F.mse_loss(flat(self.f_adv[self.domain_mask == 1]),
        #                               flat(self.f_seq[self.domain_mask == 1]))
        # loss_ADV_tgt = - F.mse_loss(flat(self.f_adv[self.domain_mask == 0]),
        #                               flat(self.f_seq[self.domain_mask == 0]))
        # self.loss_D = (loss_ADV_src * self.lambda_src + loss_ADV_tgt * self.lambda_tgt) / (self.lambda_src + self.lambda_tgt)
        
        self.optimizer_D.step()
        return loss_ADV_src.item(), loss_ADV_tgt.item()

    def __optimize_EF__(self):
        self.netD.eval(), self.netG.eval()
        self.netE.train(), self.netF.train(), # self.netF_value.train()

        self.optimizer_EF.zero_grad()
        self.loss_E_pred = F.nll_loss(flat(self.f_seq[self.domain_mask == 1]),
                                      flat(self.y_seq[self.domain_mask == 1]))

        self.f_adv, self.f_adv_softmax = self.netD(self.e_seq, return_softmax=True)
        self.loss_ADV_src = F.nll_loss(flat(self.f_adv[self.domain_mask == 1]),
                                       flat(self.g_seq[self.domain_mask == 1]))
        f_adv_tgt = torch.log(1 - self.f_adv_softmax[self.domain_mask == 0] + 1e-10)
        self.loss_ADV_tgt = F.nll_loss(flat(f_adv_tgt),
                                       flat(self.g_seq[self.domain_mask == 0]))
        self.loss_E_adv = -(self.loss_ADV_src * self.lambda_src + self.loss_ADV_tgt * self.lambda_tgt) / (self.lambda_src + self.lambda_tgt)
        self.loss_E = self.loss_E_pred + self.opt.lambda_gan * self.loss_E_adv

        self.loss_E.backward()

        self.optimizer_EF.step()

        return self.loss_E_pred.item(), self.loss_E_adv.item() # loss_E_pred_value.item()




class GDA(BaseModel):
    """
    GDA Model
    """
    def __init__(self, opt):
        super(GDA, self).__init__(opt)
        # all the necessary net
        # self.netE = GraphSeqFeatureNet(opt).to(opt.device)
        # self.netF = GraphSeqPredNet(opt).to(opt.device)
        
        self.netE = FeatureNet(opt).to(opt.device)
        # self.netE = AlexNet_BVLC_Feature(opt).to(opt.device) # GRUSeqFeatureNet(opt).to(opt.device)
        self.netF = PredNet(opt).to(opt.device) # GRUSeqPredNet(opt).to(opt.device)
        # self.netF_value = GRUSeqPredValueNet(opt).to(opt.device)
        # self.netE = FeatureNet(opt).to(opt.device)
        # self.netF = PredNet(opt).to(opt.device)

        self.netG = GNet(opt).to(opt.device)
        self.netD = GraphDNet(opt).to(opt.device)
        # self.netD = DenseDNet(opt).to(opt.device)
        self.__init_weight__()

        # after initialization, load old weights!!
        # TODO: warm start for feature for all the code
        # if opt.use_pretrain_f:
        #     self.netE.load_state_dict(torch.load(opt.feature_path), strict=False)

        EF_parameters = list(self.netE.parameters()) + list(self.netF.parameters()) # + list(self.netF_value.parameters())
        self.optimizer_EF= optim.Adam(EF_parameters, lr=opt.lr_e, betas=(opt.beta1, 0.999))
        self.optimizer_D = optim.Adam(self.netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
        # temporary just try the 
        if not self.use_g_encode:
            self.optimizer_G =  optim.Adam(self.netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))

        self.lr_scheduler_EF = lr_scheduler.ExponentialLR(optimizer=self.optimizer_EF, gamma=0.5 ** (1/100))
        self.lr_scheduler_D = lr_scheduler.ExponentialLR(optimizer=self.optimizer_D, gamma=0.5 ** (1/100))
        if not self.use_g_encode:
            self.lr_scheduler_G = lr_scheduler.ExponentialLR(optimizer=self.optimizer_G, gamma=0.5 ** (1 / 100))
            self.lr_schedulers = [self.lr_scheduler_EF, self.lr_scheduler_D, self.lr_scheduler_G]
        else:
            self.lr_schedulers = [self.lr_scheduler_EF, self.lr_scheduler_D]
        self.loss_names = ['E_pred','E_gan', 'D', 'G'] # 'E_pred_value',, 'E_concen'
        if self.use_g_encode:
            self.loss_names.remove('G')


    def __train_forward__(self):
        # TODO: implement the net
        self.z_seq = self.netG(self.one_hot_seq)        # be sure about the one_hot_seq meaning!
        # self.e_seq = self.netE(self.x_seq, self.z_seq)  # encoder of the data
        self.e_seq = self.netE(self.x_seq, self.z_seq)  # encoder of the data
        self.f_seq = self.netF(self.e_seq)              # regression for prediction
        # self.f_value_seq = self.netF_value(self.e_seq)
        # self.d_seq = self.netD(self.e_seq)

        if self.opt.lambda_gan != 0:
            self.d_seq = self.netD(self.e_seq)

            # this is the d loss, still not backward yet
            self.loss_D = self.__loss_D__(self.d_seq)

    def __test_forward__(self):
        self.z_seq = self.netG(self.one_hot_seq)        # be sure about the one_hot_seq meaning!
        self.e_seq = self.netE(self.x_seq, self.z_seq)  # encoder of the data
        self.f_seq = self.netF(self.e_seq)
        # self.f_value_seq = self.netF_value(self.e_seq)   
        self.g_seq = torch.argmax(self.f_seq.detach(), dim=2)
        # self.g_seq = torch.argmax(self.f_seq.detach(), dim=2)

    def __optimize__(self):
        loss_value = dict()
        if not self.use_g_encode:
            loss_value['G'] = self.__optimize_G__()
        if self.opt.lambda_gan != 0:
            loss_value['D'] = self.__optimize_D__()
        else:
            loss_value['D'] = 0
        # print(loss_value['D'])
        loss_value['E_pred'], loss_value['E_gan'] = self.__optimize_EF__() # loss_value['E_pred_value'], 
        # loss_value['E_pred'], loss_value['E_gan'], loss_value['E_concen'] = self.__optimize_EF__() # loss_value['E_pred_value'], 
        return loss_value

    def __optimize_G__(self):
        self.netG.train()
        self.netD.eval(), self.netE.eval(), self.netF.eval(), # self.netF_value.eval()

        self.optimizer_G.zero_grad()

        criterion = nn.BCEWithLogitsLoss()

        sub_graph = self.__sub_graph__(my_sample_v=self.opt.sample_v_g)
        errorG = torch.zeros((1,)).to(self.device) # .double()
        sample_v = self.opt.sample_v_g
        # train_z_seq = self.z_seq.reshape()
        
        for i in range(sample_v):
            v_i = sub_graph[i]
            for j in range(i + 1, sample_v):
                v_j = sub_graph[j]
                # label = torch.tensor(self.opt.A[v_i][v_j]).to(self.device)
                # label = torch.full((self.batch_size,), self.opt.A[v_i][v_j], device=self.device)
                label = torch.tensor(self.opt.A[v_i][v_j]).to(self.device) # , dtype=torch.double
                # dot product
                # output = self.netG.weight * (self.z_seq[v_i * self.batch_size] * self.z_seq[v_j * self.batch_size]).sum() + self.netG.bias
                output = (self.z_seq[v_i * self.tmp_batch_size] * self.z_seq[v_j * self.tmp_batch_size]).sum()
                errorG += criterion(output, label)

        errorG /= (sample_v * (sample_v - 1) / 2)

        errorG.backward(retain_graph=True)
        
        self.optimizer_G.step()
        return errorG.item()

    def __optimize_D__(self):
        self.netD.train()
        self.netG.eval(), self.netE.eval(), self.netF.eval(), # self.netF_value.eval()

        self.optimizer_D.zero_grad()

        # backward process:
        self.loss_D.backward(retain_graph=True)

        self.optimizer_D.step()
        return self.loss_D.item()

    def __loss_D__(self, d):
        criterion = nn.BCEWithLogitsLoss()
        # criterion = nn.MSELoss()

        # random pick subchain and optimize the D
        # balance coefficient is calculate by pos/neg ratio
        # print("I am here")
        sub_graph = self.__sub_graph__(my_sample_v=self.opt.sample_v)

        errorD_connected = torch.zeros((1,)).to(self.device)  # .double()
        errorD_disconnected = torch.zeros((1,)).to(self.device) # .double()

        count_connected = 0
        count_disconnected = 0

        # print("=============")
        # print(sub_graph)
        # print(d.shape)
        
        for i in range(self.opt.sample_v):
            # be careful!!
            if self.opt.sample_neighbour and i >= len(sub_graph):
                break
            v_i = sub_graph[i]
            # for j in range(i, self.opt.sample_v):
            # !! debug, no self loop version!!
            for j in range(i + 1, self.opt.sample_v):
                if self.opt.sample_neighbour and j >= len(sub_graph):
                    break
                
                v_j = sub_graph[j]
                
                # TODO!!!! for seq len data only!!
                label = torch.full((self.tmp_batch_size,), self.opt.A[v_i][v_j], device=self.device) # , dtype=torch.double
                # dot product
                if v_i == v_j:
                    # be careful about the index range!
                    idx = torch.randperm(self.tmp_batch_size)
                    output = (d[v_i][idx] * d[v_j]).sum(1)
                    # print(output)
                else:
                    output = (d[v_i] * d[v_j]).sum(1)
                    # print(output)

                # print("==========")
                # print(output)
                # try
                # output = torch.clamp(output, 0, 1)

                if self.opt.A[v_i][v_j]: # connected
                    # print("=========")
                    # print(output.shape)
                    # print(label.shape)
                    errorD_connected += criterion(output, label)
                    count_connected += 1
                else:
                    errorD_disconnected += criterion(output, label)
                    count_disconnected += 1

        # all_error_count = count_disconnected + count_connected
        errorD = 0.5 * (errorD_connected / count_connected + errorD_disconnected / count_disconnected)
        # print(errorD.item())
        # this is a loss balance
        return errorD * self.num_domain
        # return errorD

    # def __optimize_EF__(self):
    #     self.netD.eval(), self.netG.eval()
    #     self.netE.train(), self.netF.train(), # self.netF_value.train()

    #     self.optimizer_EF.zero_grad()
    #     # criterion = nn.MSELoss()
        
    #     if self.opt.lambda_gan != 0:
    #         loss_E_gan = - self.loss_D
    #     else:
    #         loss_E_gan = torch.tensor(0, dtype=torch.double, device=self.opt.device)

    #     y_seq_source = self.y_seq[self.domain_mask == 1]
    #     f_seq_source = self.f_seq[self.domain_mask == 1]

    #     # y_value_seq_source = self.y_value_seq[self.domain_mask == 1]
    #     # f_value_seq_source = self.f_value_seq[self.domain_mask == 1]

    #     # if (self.epoch + 1) % 40 == 0:
    #     #     print("++++++++++++++")
    #     #     print(self.netE.fc_final.weight)
    #     #     print(self.netF.linear.weight)
    #     # print(flat(f_seq_source))
    #     # print("==============")
    #     # print(flat(y_seq_source))
    #     # loss_E_pred = F.l1_loss(flat(f_seq_source), flat(y_seq_source))
    #     # print(f_seq_source.shape)
    #     # print(y_seq_source.shape)
    #     loss_E_pred = F.nll_loss(flat(f_seq_source), flat(y_seq_source))
    #     # loss_E_pred = nn.BCEWithLogitsLoss()(flat(f_seq_source), flat(y_seq_source))
    #     # print(f_seq_source)
    #     # print("============")
    #     # print(y_seq_source)
    #     # loss_E_pred = F.nll_loss(flat(f_seq_source), flat(y_seq_source))

    #     # loss_E_pred_value = F.mse_loss(flat(f_value_seq_source), flat(y_value_seq_source))

    #     # loss_E_pred_all = self.opt.lambda_reg * loss_E_pred_value + loss_E_pred

    #     # add concentrate loss
    #     # domain * batch * (seq * embed dim)
    #     # if self.opt.concentrate_on_all_domain:
    #     #     self.e_seq_mean = torch.mean(self.e_seq, dim=(0,1), keepdim=True)
    #     # else:
    #     #     self.e_seq_mean = torch.mean(self.e_seq, dim=1, keepdim=True)
    #     # loss_E_concentrate = ((self.e_seq - self.e_seq_mean) ** 2).sum()

    #     loss_E = loss_E_gan * self.opt.lambda_gan + loss_E_pred # + self.opt.lambda_concen * loss_E_concentrate
    #     loss_E.backward()

    #     self.optimizer_EF.step()

    #     return loss_E_pred.item(), loss_E_gan.item(), loss_E_concentrate.item() # loss_E_pred_value.item(), 

