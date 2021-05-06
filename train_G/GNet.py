import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: G now is a fixed embedding:
class GNet(nn.Module):
    def __init__(self, opt):
        super(GNet, self).__init__()
        # for cluster
        # self.G = torch.FloatTensor(
        #     [
        #         [1, 0],
        #         [1, 1],
        #         [0, 1],
        #     ]
        # ).to(opt.device)

        # for quarter
        # num_domain = opt.num_domain
        # G = np.eye(num_domain, num_domain - 1) + np.eye(num_domain, num_domain - 1, k=-1)
        # self.G = torch.from_numpy(G).float().to(device=opt.device)

        # the following code is for testing G only:


        # self.G = torch.randn(opt.num_domain, opt.nt).to(opt.device)
        # self.G.requires_grad=True

        self.fc1 = nn.Linear(opt.num_domain, opt.nh)
        # self.fc2 = nn.Linear(opt.nh, opt.nh)
        self.fc_final = nn.Linear(opt.nh, opt.nt)


    def forward(self, x):
        # return torch.matmul(x.float(), self.G)
        # drop out:
        # p = self.opt.p
        # x = nn.Dropout(p=p)(x.float())
        x = F.relu(self.fc1(x.float()))
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = nn.Dropout(p=p)(x)
        x = self.fc_final(x)
        return x