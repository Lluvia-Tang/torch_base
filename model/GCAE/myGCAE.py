# -*- coding:utf-8 -*-
__author__ = 'tangs'
__time__ = '2021/12/1 14:56'

import torch
from torch import nn
import torch.nn.functional as F


class GCAE(nn.Module):

    def __init__(self, input_size, embedding_dim, n_filters, kernel_size, weights=None, net_dropout=0.5):

        super(GCAE, self).__init__()
        self.in_channels = 1
        self.out_channels = 3
        self.model_name = 'GCAE_Model'

        self.dropout = nn.Dropout(net_dropout)
        if weights is not None:
            self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_dim, _weight=weights)
        else:
            self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_dim)
        self.k = 1

        self.linear = nn.Linear(len(kernel_size) * n_filters, input_size)

        self.fc = nn.Linear(len(kernel_size) * n_filters, 3)
        self.out = nn.Linear(input_size, 3)  # 3分类
        self.relu = nn.ReLU()

        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1,
                                              out_channels=n_filters, kernel_size=(K, embedding_dim))
                                    for K in kernel_size])

        self.convs_tanh = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(K, embedding_dim)) for K in kernel_size])
        self.convs_relu = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(K, embedding_dim)) for K in kernel_size])
        self.V = nn.Parameter(torch.rand([embedding_dim, n_filters], requires_grad=True).cuda())

    def forward(self, x):
        # x=[batch_size, seq_len]
        text_indices, target = x[0],x[2]
        batch_size, seq_len = text_indices.shape
        # print(text_indices)
        # print("target shape:+++++++++++++\n",target.shape) [100,1]
        target = target[1]

        embedded = self.embedding(text_indices)
        # embedded = [batchsize, seq_len, embedding_dim]

        embedded = embedded.unsqueeze(1)
        # print(embedded.shape)
        # embedded = [batchsize, (in_channel)1, seq_len, embedding_dim]

        # a = torch.zeros(batch_size, 4)
        # a = a.long()
        # for i in range(batch_size):
        #     a[i, :] = target
        # target = a.t()
        # target = a.to("cuda:0")
        t_emb = self.embedding(target)
        # print(t_emb.shape)
        t_emb = torch.mean(t_emb, dim=0, keepdim=True)  # ([ 1, embedding_dim])
        # print(t_emb.shape)  #[1,1,300]

        conved_tanh = [F.tanh(conv(embedded).squeeze(3)) for conv in self.convs_tanh]
        # conved_tanh = [batchsize, n_filters*out_channels, text_len-filter_size[n] +1]
        # conved = [self.relu(conv(embedded).squeeze(3)) for conv in self.convs]

        # print(torch.mm(t_emb, self.V).unsqueeze(2).shape)
        #[1, n_filters,1]

        conved_relu = [self.relu(conv(embedded).squeeze(3) + torch.mm(t_emb, self.V).unsqueeze(2)) for conv in
                       self.convs_relu]
        # print(conved_relu[0].shape) [batchsize, n_filters, text_len]

        conved_mul = [i * j for i, j in zip(conved_tanh, conved_relu)]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved_mul]

        cat = self.dropout(torch.cat(pooled, dim=1))

        linear = self.relu(self.linear(cat))
        linear = self.dropout(linear)

        out = self.out(linear)

        return out