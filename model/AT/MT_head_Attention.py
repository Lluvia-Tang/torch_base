# -*- coding:utf-8 -*-
__author__ = 'tangs'
__time__ = '2021/12/17 14:19'
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

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
        #此时输入没有seqlen，如果有的话自行将n=1修改为
        # batch,n, dim_in = x.shape

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