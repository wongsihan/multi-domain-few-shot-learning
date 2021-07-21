#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.nn.utils.weight_norm import WeightNorm
from torch.nn.modules import Module


class CNNEncoder2d(nn.Module):
    """docstring for ClassName"""

    def __init__(self, feature_dim):
        super(CNNEncoder2d, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim, momentum=1, affine=True),
            nn.ReLU())
        self.admp = nn.AdaptiveMaxPool2d((5, 5))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.admp(out)
        # out = out.view(out.size(0),-1)
        return out  # 64


class RelationNetwork2d(nn.Module):
    """docstring for RelationNetwork"""

    def __init__(self, input_size, hidden_size):
        super(RelationNetwork2d, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_size * 2, input_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_size, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(input_size, input_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_size, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out


class CNNEncoder1d(nn.Module):
    """docstring for ClassName"""

    def __init__(self, feature_dim):
        super(CNNEncoder1d, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=10, padding=0, stride=3),
            nn.BatchNorm1d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm1d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64, momentum=1, affine=True),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv1d(64, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim, momentum=1, affine=True),
            nn.ReLU())
        self.admp = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.admp(out)
        # out = out.view(out.size(0),-1)
        return out  # 64


class RelationNetwork1d(nn.Module):
    """docstring for RelationNetwork"""

    def __init__(self, input_size, hidden_size):
        num = int(input_size / 2)
        super(RelationNetwork1d, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(input_size * 2, input_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(input_size, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(input_size, input_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(input_size, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2))

        self.fc1 = nn.Linear(input_size * 6, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


class Classifier1d(nn.Module):
    """docstring for ClassName"""

    def __init__(self, input_size, out_size):
        super(Classifier1d, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(input_size, input_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(input_size, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(input_size, input_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(input_size, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.fc1 = nn.Linear(input_size * 6, out_size)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out


class CNN1d(nn.Module):
    """docstring for ClassName"""

    def __init__(self, feature_dim):
        super(CNN1d, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=10, padding=0, stride=3),
            nn.BatchNorm1d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm1d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64, momentum=1, affine=True),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv1d(64, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim, momentum=1, affine=True),
            nn.ReLU())
        self.admp = nn.AdaptiveMaxPool1d(25)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        # outv1 = self.conv1(out)
        #
        # r = out - outv1
        # r = self.avg_pool(r)
        # r = self.fc1(r)
        # r = self.relu(r)
        # r = self.fc2(r)
        # r = self.sigmoid(r)
        # a_vec = self.sigmoid(r).expand(x.size(0), 64, 83)
        # r_plus = torch.mul(a_vec, r)
        # out = outv1 + r_plus
        out = self.layer4(out)
        out = self.admp(out)
        out = out.view(out.size(0),-1)
        return out


class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear( indim, outdim, bias = False)
        self.class_wise_learnable_norm = True  #See the issue#4&8 in the github
        if self.class_wise_learnable_norm:
            WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm

        if outdim <=200:
            self.scale_factor = 2; #a fixed scale factor to scale the output of cos value into a reasonably large input for softmax, for to reproduce the result of CUB with ResNet10, use 4. see the issue#31 in the github
        else:
            self.scale_factor = 10; #in omniglot, a larger scale factor is required to handle >1000 output classes.

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+ 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized) #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor* (cos_dist)

        return scores

def conv_block(in_channels, out_channels):
    '''
    returns a block conv-bn-relu-pool
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class ProtoNet(nn.Module):
    '''
    Model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    '''
    def __init__(self, x_dim=1, hid_dim=64, z_dim=64):
        super(ProtoNet, self).__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

class PrototypicalLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''
    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        return prototypical_loss(input, target, self.n_support)

def prototypical_loss(input, target, n_support):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py

    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')

    def supp_idxs(c):
        # FIXME when torch will support where as np
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    # FIXME when torch.unique will be available on cuda too
    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    # FIXME when torch will support where as np
    # assuming n_query, n_target constants
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

    support_idxs = list(map(supp_idxs, classes))

    #prototypes是原型的特征(60,64)
    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
    # FIXME when torch will support where as np
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)

    query_samples = input.to('cpu')[query_idxs]#(300,64)
    dists = euclidean_dist(query_samples, prototypes)#(300,60)

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)#60,5,60

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()#(60,5,1)
    target_inds = target_inds.to('cpu')

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()#(60,5).view(-1).mean()
    _, y_hat = log_p_y.max(2)#yhat(60,5)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

    return loss_val,  acc_val

def euclidean_dist(x, y):#x(300,64) y(60,64)
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)#300
    m = y.size(0)#60
    d = x.size(1)#64
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)