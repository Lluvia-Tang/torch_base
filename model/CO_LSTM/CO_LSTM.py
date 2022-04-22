# -*- coding:utf-8 -*-
__author__ = 'tangs'
__time__ = '2021/12/16 21:02'

import torch
from torch import nn
import torch.nn.functional as F

class CoLSTM(nn.Module):
    def __init__(self,input_size, n_filters, kernel_size, embedding_dim, hidden_size, weights=None, dropout=0.5):
        super(CoLSTM,self).__init__()
        self.cnn = KimCNN(input_size,embedding_dim,n_filters,kernel_size,weights)
        self.bilstm = BiLSTM(1, hidden_size, dropout=0.5)

    def forward(self, x):
        cnn_out = self.cnn(x)
        # [batchsize, n_filters * len(filter_sizes)]
        out = self.bilstm(cnn_out)
        return out


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]

    return x.gather(dim, index)


class KimCNN(nn.Module):

    def __init__(self, input_size, embedding_dim, n_filters, kernel_size, weights=None, net_dropout=0.5):

        super(KimCNN, self).__init__()
        self.in_channels = 1
        self.out_channels = 1
        self.model_name = 'Kim_CNN'

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

        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1,
                                              out_channels=n_filters, kernel_size=(K, embedding_dim))
                                    for K in kernel_size])

    def forward(self, x):
        # x[0]=[batch_size, seq_len]
        x = x[0]
        batch_size, seq_len = x.shape

        embedded = self.embedding(x)
        # embedded = [batchsize, seq_len, embedding_dim]

        embedded = embedded.unsqueeze(1)
        # print(embedded.shape)
        # embedded = [batchsize, (in_channel)1, seq_len, embedding_dim]

        conved = [self.relu(conv(embedded).squeeze(3)) for conv in self.convs]
        # conved = [batchsize, n_filters*out_channels, text len-filter_size[n] +1]

        # pooled = [kmax_pooling(i, 2, self.k).view(-1,out_channels*self.k) for i in conved]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled = [batchsize, n_filters*outchannels]


        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = [batchsize, n_filters * len(filter_sizes)]

        return cat


class BiLSTM(nn.Module):

    def __init__(self, embed_dim, hidden_size, dropout=0.5):

        super(BiLSTM, self).__init__()


        self.hidden_size = hidden_size
        self.dropout = dropout
        self.n_layers = 1


        self.lstm = nn.LSTM(embed_dim, hidden_size, dropout=dropout, bidirectional=True)
        self.fc = torch.nn.Linear(hidden_size * 2, 3)  # 分成3类
        # self.linear = nn.Linear(self.hidden_size*2, linear_size)
        # self.out = nn.Linear(linear_size, 3)
        # self.relu = nn.ReLU()

    def forward(self, input):
        #input = [batchsize, n_filters * len(filter_sizes)]
        batch_size, inputsize = input.shape  #[64,48]
        # print(batch_size,inputsize)
        # hidden = self._init_hidden(batch_size)
        input = input.unsqueeze(2)  #[batchsize, n_filters * len(filter_sizes), 1] flatten到LSTM中
        input = input.transpose(0,1)    #[seq,batchsize,1]
        h0 = torch.randn(self.n_layers * 2, batch_size, self.hidden_size)
        h0 = h0.to(device="cuda:0")
        c0 = torch.randn(self.n_layers * 2, batch_size, self.hidden_size)
        c0 = c0.to(device="cuda:0")

        # print("h0",h0.shape)
        output, (_, _) = self.lstm(input, (h0, c0))  # [batch_size, hidden_size*2]
        output = output[-1]  # [batch_size, hidden_size*2] 只需要最后一个输出
        # hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        fc_output = self.fc(output)
        # fc_output = self.fc(hidden_cat)
        return fc_output