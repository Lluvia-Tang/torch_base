# -*- coding:utf-8 -*-
__author__ = 'tangs'
__time__ = '2021/12/16 17:51'
'''
multi-channel CNN + multi-head attention
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import random

#单个channel模型
class single_Channel_CNN(nn.Module):
    def __init__(self, vocab_size, au_vocab_size,embedding_dim, au_embedding_dim, hidden_dim,
                 n_filters, kernel_size, embed_weights=None, au_weight=None, net_dropout=0.5):
        super(single_Channel_CNN, self).__init__()
        self.relu = nn.ReLU()
        self.in_channels = 1
        self.dropout = nn.Dropout(net_dropout)
        # 词嵌入
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, _weight=embed_weights)
        # 其他嵌入
        self.au_embeddings = nn.Embedding(au_vocab_size, au_embedding_dim, _weight=au_weight)

        self.k = 1
        # self.linear = nn.Linear(self.out_channels*4, input_size)
        # self.linear = nn.Linear(len(kernel_size) * n_filters, input_size)

        # self.fc = nn.Linear(len(kernel_size) * n_filters, 3)
        # self.out = nn.Linear(input_size, 3)  # 3分类
        # self.relu = nn.ReLU()

        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1,
                                              out_channels=n_filters, kernel_size=(K, embedding_dim+au_embedding_dim))
                                    for K in kernel_size])

        # # 2 convs

        # in_fea = len(self.kernel_size)*self.kernel_num
        # self.linear1 = nn.Linear(in_fea, in_fea // 2)
        # self.linear2 = nn.Linear(in_fea // 2, self.label_num)
        # self.embed_dropout = nn.Dropout(self.embed_dropout)
        # self.fc_dropout = nn.Dropout(self.fc_dropout)

    def forward(self, text, au): #第几个channel

        text_indices, au_indices = text, au

        au_embed = self.au_embeddings(au_indices)  #[batchsize, seq_len, 100]
        embed = self.word_embeddings(text_indices)   #[batchsize, seq_len, 300]
        #concat词嵌入
        xa_emb = torch.cat((embed, au_embed), dim=2) #[batchsize, seq_len, 400]

        embedded = xa_emb.unsqueeze(1)
        # embedded = [batchsize, (in_channel)1, seq_len, embedding_dim]

        conved = [self.relu(conv(embedded).squeeze(3)) for conv in self.convs]
        # conved = [batchsize, n_filters*out_channels, text len-filter_size[n] +1]

        # pooled = [kmax_pooling(i, 2, self.k).view(-1,out_channels*self.k) for i in conved]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled = [batchsize, n_filters*outchannels]

        return pooled

class Multi_Channel_CNN(nn.Module):
    def __init__(self, vocab_size, au_vocab_size_list,embedding_dim, hidden_dim,
                 n_filters, kernel_size, embed_weights=None, au_weight=None, net_dropout=0.5):
        super(Multi_Channel_CNN, self).__init__()
        self.in_channel = 2
        self.au_vocab_size_list = au_vocab_size_list
        self.au_embedding_dim = 100
        self.tag_vocab_size = au_vocab_size_list[0]
        self.position_vocab_size = au_vocab_size_list[1]
        self.tag_embed_weight = au_weight[0]
        self.position_embed_Weight = au_weight[1]
        self.tag_pooled = single_Channel_CNN(vocab_size, self.tag_vocab_size,embedding_dim, self.au_embedding_dim, hidden_dim,
                 n_filters, kernel_size, embed_weights=embed_weights, au_weight=self.tag_embed_weight, net_dropout=0.5)
        self.position_pooled = single_Channel_CNN(vocab_size, self.position_vocab_size,embedding_dim, self.au_embedding_dim, hidden_dim,
                 n_filters, kernel_size, embed_weights=embed_weights, au_weight=self.position_embed_Weight, net_dropout=0.5)

        self.linear = nn.Linear(self.in_channel* len(kernel_size) * n_filters, vocab_size)

        self.fc = nn.Linear(len(kernel_size) * n_filters, 3)
        self.out = nn.Linear(vocab_size, 3)  # 3分类
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(net_dropout)
        self.dim_in = self.in_channel * n_filters * len(kernel_size)
        self.mt_at = Multi_head_Attention(self.dim_in, self.dim_in, self.dim_in, n_head=8)

    def forward(self, input):
        text = input[0]
        # print(input.shape)
        tag, position = input[3],input[4]
        tag_pooled = self.tag_pooled(text, tag)
        position_pooled = self.position_pooled(text, position)
        # pooled = [batchsize, n_filters]

        tag_cat = self.dropout(torch.cat(tag_pooled, dim=1))
        position_cat = self.dropout(torch.cat(position_pooled, dim=1))
        # cat = [batchsize, n_filters * len(filter_sizes)]
        att_input = torch.cat((tag_cat, position_cat),dim=1)
        # [batchsize, 2 * n_filters * len(filter_sizes)]
        mh_att_out = self.mt_at(att_input)
        linear = self.relu(self.linear(mh_att_out))
        linear = self.dropout(linear)

        out = self.out(linear)

        return out


class Multi_head_Attention(nn.Module):
    dim_in: int  # input dimension
    dim_k: int  # key and query dimension
    dim_v: int  # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads
    def __init__(self, dim_in, dim_k, dim_v, n_head=8):
        super(Multi_head_Attention, self).__init__()
        assert dim_k % n_head == 0 and dim_v % n_head == 0, "dim_k and dim_v must be multiple of num_heads"
        # self.size_per_head = size_per_head  # 每个头的大小 原始传入向量长度/头个数
        # self.output_dim = n_head * size_per_head  # 输出维度，所有头拼起来
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = n_head
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k // self.num_heads)

    def forward(self, x):
        # x: tensor of shape (batch, n, dim_in)
        #x [batch, dim]
        batch, dim_in = x.shape
        assert dim_in == self.dim_in

        x = x.unsqueeze(1)
        n = 1
        # x: [batch, n(1),dim_in]
        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head

        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n

        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v
        att = att.squeeze(1)  # batch, dim_v

        return att