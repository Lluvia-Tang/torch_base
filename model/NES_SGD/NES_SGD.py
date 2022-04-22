# -*- coding:utf-8 -*-
__author__ = 'tangs'
__time__ = '2021/12/25 13:03'

'''
Learned Weight Sharing for Deep Multi-Task Learning by Natural Evolution Strategy and Stochastic Gradient Descent
'''

import torch
from torch import nn
import torch.nn.functional as F
from model.NES_SGD.task_routing_model import StaticTaskRouting, LearnedTaskRouting, IgnoreTaskRouting
from utils.torch_utils import Flatten

def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]

    return x.gather(dim, index)

class ResidualBlock(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, (3,300),  padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
            nn.Conv2d(ch_out, ch_out, (3,300),  padding=1, bias=False),
            nn.BatchNorm2d(ch_out)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or ch_in != ch_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, (1,300) , bias=False),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        x = x[0]
        batch_size, seq_len = x.shape

        embedded = self.embedding(x)
        # embedded = [batchsize, seq_len, embedding_dim]
        embedded = embedded.unsqueeze(1)
        out = self.main(embedded)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class KimCNN(nn.Module):

    def __init__(self,task_outputs ,input_size, embedding_dim, kernel_size, num_modules, weights=None, n_filters=64, net_dropout=0.5):

        super(KimCNN, self).__init__()
        self.in_channels = 1
        self.out_channels = 1
        self.num_tasks = 3
        self.num_modules = num_modules #共享权重集合数量=20
        self.k = kernel_size  #=3


        self.dropout = nn.Dropout(net_dropout)
        if weights is not None:
            self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_dim, _weight=weights)
        else:
            self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_dim)
        self.k = 1
        # self.linear = nn.Linear(self.out_channels*4, input_size)
        self.linear = nn.Linear(len(kernel_size) * n_filters, input_size)

        self.fc = nn.Linear(len(kernel_size) * n_filters, 3)
        self.out = nn.Linear(input_size, 3)  # 3分类
        self.relu = nn.ReLU()
        # print(input_size)

        # print("+++++input_size:+++++\n", input_size)
        # self.convs = nn.ModuleList([nn.Conv2d(in_channels=1,
        #                                       out_channels=out_channels,kernel_size=K)
        #                             for K in kernel_size])

        # conv: [input_channel(=1), output_channel, (filter_height, filter_width), stride=1]
        # self.convs = nn.ModuleList([nn.Conv2d(in_channels=1,
        #                                       out_channels=n_filters, kernel_size=(K, embedding_dim))
        #                             for K in kernel_size])

        self.conv1 = IgnoreTaskRouting(nn.Sequential(
            nn.Conv2d(1, out_channels=n_filters, kernel_size=(self.k,embedding_dim), bias=False),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
        ))

        self.layer1 = self.make_layer(64, 2, stride=2)
        self.layer2 = self.make_layer(128, 2, stride=2)
        self.layer3 = self.make_layer(256, 2, stride=2)
        self.layer4 = self.make_layer(512, 2, stride=2)
        self.pool_flatten = IgnoreTaskRouting(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten()
        ))
        self.fc = StaticTaskRouting(self.num_tasks, [nn.Linear(512, s) for s in task_outputs])

    def make_layer(self, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            block = self.make_wrapped_block(channels, stride)
            layers.append(block)
            self.ch_in = channels
        return nn.Sequential(*layers)

    def make_wrapped_block(self, channels, stride):
        raise NotImplementedError

    def make_block(self, channels, stride):
        return ResidualBlock(self.ch_in, channels, stride)

    def forward(self, x):
        # x=[batch_size, seq_len]
        x = x[0]
        batch_size, seq_len = x.shape

        embedded = self.embedding(x)
        # embedded = [batchsize, seq_len, embedding_dim]

        embedded = embedded.unsqueeze(1)
        # print(embedded.shape)
        # embedded = [batchsize, (in_channel)1, seq_len, embedding_dim]

        out = self.conv1(embedded)
        # conved = [batchsize, n_filters*out_channels, text len-filter_size[n] +1]
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool_flatten(out)
        out = self.fc(out)

        return out


class LearnedSharingResNet18(KimCNN):
    def make_wrapped_block(self, channels, stride):
        return LearnedTaskRouting(self.num_tasks, [
            self.make_block(channels, stride) for _ in range(self.num_modules)
        ])


