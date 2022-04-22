# -*- coding:utf-8 -*-
__author__ = 'tangs'
__time__ = '2021/12/22 17:42'
import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel
from math import sqrt

def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]

    return x.gather(dim, index)

class BertForClassification(nn.Module):
    def __init__(self, model,num_classes):
        super(BertForClassification,self).__init__()
        self.bert = BertModel.from_pretrained(model)
        for param in self.bert.parameters():
            param.requires_grad = True  # 使参数可更新
        # self.hidden_size = 1024  #covid-bert的embedding是1024不同于768
        self.hidden_size = 768
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        #The classification layer that takes the [CLS] representation and outputs the logit
        # self.hidden_layer = nn.Linear(model.hidden_size, model.hidden_size)
        self.fc = nn.Linear(self.hidden_size, num_classes)

    def forward(self, input):
        '''
        Inputs:
            -input_ids : Tensor of shape [B, T] containing token ids of sequences
            -attention_mask : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
            (where B is the batch size and T is the input length)
        '''
        #Feed the input to Bert model to obtain outputs
        input_ids, attention_mask = input[0],input[1]
        # outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 第一个参数 是所有输入对应的输出  第二个参数 是 cls最后接的分类层的输出
        outputs = self.bert(input_ids, attention_mask)  # output_all_encoded_layers 是否将bert中每层(12层)的都输出，false只输出最后一层【128*768】
        # print(outputs.pooler_output.shape)
        # embeded = outputs.pooler_output  # (batch_size, hidden_size) hidden_size=768
        embeded = outputs.last_hidden_state  # (batch_size, sequence_length, hidden_size) hidden_size=768
        # print("out:",out)
        return embeded


class StanceCNN(nn.Module):

    def __init__(self,model, n_filters, kernel_size, net_dropout=0.5):

        super(StanceCNN, self).__init__()
        self.in_channels = 1
        self.out_channels = 3
        embedding_dim = 768

        self.dropout = nn.Dropout(net_dropout)
        self.k = 1
        # self.linear = nn.Linear(self.out_channels*4, input_size)
        # self.linear = nn.Linear(len(kernel_size) * n_filters, input_size)

        self.fc = nn.Linear(len(kernel_size) * n_filters, 3)
        self.out = nn.Linear(len(kernel_size) * n_filters, 3)  # 3分类
        self.relu = nn.ReLU()
        self.bert = BertForClassification(model,3)
        self.attention = Multi_head_Attention(768,768,768)
        # print(input_size)

        # print("+++++input_size:+++++\n", input_size)
        # self.convs = nn.ModuleList([nn.Conv2d(in_channels=1,
        #                                       out_channels=out_channels,kernel_size=K)
        #                             for K in kernel_size])

        # conv: [input_channel(=1), output_channel, (filter_height, filter_width), stride=1]
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1,
                                              out_channels=n_filters, kernel_size=(K, embedding_dim))
                                    for K in kernel_size])

    def forward(self, x):
        # x=[batch_size, seq_len]
        embedded = self.bert(x)  # (batch_size,seq, 768)
        mh_att_out = self.attention(embedded) # (batch_size,seq, 768)
        # print(embedded.shape)
        embedded = mh_att_out.unsqueeze(1)
        # embedded = [batchsize, (in_channel)1, seq_len, 768]

        conved = [self.relu(conv(embedded).squeeze(3)) for conv in self.convs]
        # conved = [batchsize, n_filters*out_channels, text len-filter_size[n] +1]

        # pooled = [kmax_pooling(i, 2, self.k).view(-1,out_channels*self.k) for i in conved]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled = [batchsize, n_filters*outchannels]

        # print(pooled)
        # print(len(pooled))
        # cat = torch.cat(pooled, dim=1)
        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = [batchsize, n_filters * len(filter_sizes)]
        # print("cat.shape: \n",cat.shape)
        linear = self.relu(cat)
        linear = self.dropout(linear)
        # out = self.fc(linear)
        out = self.out(linear)
        # out = self.fc(cat)

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
        # x: tensor of shape (batch, seq, dim_in)

        batch,n, dim_in = x.shape
        # assert dim_in == self.dim_in

        # x: [batch, n,dim_in]
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
        # att = att.squeeze(1)  # batch, dim_v

        return att