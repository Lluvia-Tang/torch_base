# -*- coding:utf-8 -*-
__author__ = 'tangs'
__time__ = '2021/12/22 14:10'

import torch
from torch import nn


from model.AT.selfAttention import Attention


class T_DAN(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, weights=None, dropout = 0.5):
        super(T_DAN, self).__init__()
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.n_layers = 1
        self.tarlstm = BiLSTM(input_size,embedding_dim,hidden_size,weights)
        self.textlstm = DC_BiLSTM(input_size,embedding_dim,hidden_size,weights)

        if weights is not None:
            self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_dim, _weight=weights)
        else:
            self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_dim)

        self.attention = Attention(2 * hidden_size, 2 * hidden_size, 2 * hidden_size)
        self.fc = torch.nn.Linear(hidden_size * 2, 3)  #two-step-2分类

    def forward(self,input):
        target_out, tar_hidden = self.tarlstm(input)  #[seq, batch, hidden*2]
        tar_hidden = torch.cat([tar_hidden[-1], tar_hidden[-2]], dim=1)  # need to concat the last 2 hidden layers (bi-)
        text_out, text_hidden = self.textlstm(input)  #hidden [self.n_layers * 2, batch_size, self.hidden_size]
        text_hidden = torch.cat([text_hidden[-1], text_hidden[-2]], dim=1)
        # out = torch.cat((target_out,text_out),dim=0)  #[seq,batch, hidden*2]
        hidden = tar_hidden.mul(text_hidden) #[batch, hidden*2]

        # attention Outputs = a:[seq x batch], lin_comb:[batch x hidden*2]
        energy, atte_out = self.attention(hidden,text_out,text_out)
        # print(atte_out.shape)  [64, hidden*2]
        output = self.fc(self.dropout(atte_out))

        return output


class DC_BiLSTM(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, weights=None, dropout=0.2):

        super(DC_BiLSTM, self).__init__()

        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.n_layers = 1

        # self.hidden_size = lstm_hidden_size
        if weights is not None:
            self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_dim, _weight=weights)
        else:
            self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_dim)
        # self.embedding = torch.nn.Embedding(input_size, hidden_size)

        self.lstm = nn.LSTM(embedding_dim, hidden_size, dropout=dropout, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_size*2+embedding_dim, hidden_size, dropout=0.2, bidirectional=True)
        # self.fc = torch.nn.Linear(hidden_size * 2, 3)  # 分成3类
        # self.linear = nn.Linear(self.hidden_size*2, linear_size)
        # self.out = nn.Linear(linear_size, 3)
        # self.relu = nn.ReLU()

    def forward(self, input):

        # input[0] shape : B x S -> S x B
        # print(input)
        input = input[0]
        batch_size, seq_len = input.shape
        input = input.t()

        # batch_size = input.size(1)
        # hidden = self._init_hidden(batch_size)
        h0 = torch.randn(self.n_layers * 2, batch_size, self.hidden_size)
        h0 = h0.to(device="cuda:0")
        c0 = torch.randn(self.n_layers * 2, batch_size, self.hidden_size)
        c0 = c0.to(device="cuda:0")

        embedding = self.embedding(input)  #[seq, batch, embedding]

        output, hidden = self.lstm(embedding, (h0, c0))  # out: [seq_len, batch_size, hidden_size*2]
        output = torch.cat((output,embedding), dim=2) #[seq_len,batch,hidden_size*2+emdding]
        out,(hid,_) = self.lstm2(output,hidden)
        #out: [seq_len, batch_size, hidden_size*2]


        # out = out[-1]  # [batch_size, hidden_size*2] 只需要最后一个输出

        # fc_output = self.fc(output)
        # fc_output = self.fc(hidden_cat)
        return out,hid


class BiLSTM(nn.Module):

    def __init__(self, input_size, embedding_dim, hidden_size, weights=None, dropout=0.2):

        super(BiLSTM, self).__init__()

        self.model_name = 'BiLSTM'
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.n_layers = 1

        # self.hidden_size = lstm_hidden_size
        if weights is not None:
            self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_dim, _weight=weights)
        else:
            self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_dim)
        # self.embedding = torch.nn.Embedding(input_size, hidden_size)

        self.lstm = nn.LSTM(embedding_dim, hidden_size, dropout=dropout, bidirectional=True)
        self.fc = torch.nn.Linear(hidden_size * 2, 3)  # 分成3类
        # self.linear = nn.Linear(self.hidden_size*2, linear_size)
        # self.out = nn.Linear(linear_size, 3)
        # self.relu = nn.ReLU()

    def forward(self, input):

        # input shape : B x S -> S x B
        # print(input)
        target = input[2]
        batch_size, seq_len = target.shape
        target = target.t()

        # batch_size = input.size(1)
        # hidden = self._init_hidden(batch_size)
        h0 = torch.randn(self.n_layers * 2, batch_size, self.hidden_size)
        h0 = h0.to(device="cuda:0")
        c0 = torch.randn(self.n_layers * 2, batch_size, self.hidden_size)
        c0 = c0.to(device="cuda:0")

        embedding = self.embedding(target)

        # # pack them up
        # # lstm_input = pack_padded_sequence(embedding, seq_len)
        # output, hidden = self.lstm(embedding, hidden)
        # # if self.n_directions == 2:
        # hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        # # else:
        # #     hidden_cat = hidden[-1]
        # fc_output = self.fc(hidden_cat)
        # return fc_output

        # pack them up
        # lstm_input = pack_padded_sequence(embedding, seq_len.cpu())
        output, (hidden, _) = self.lstm(embedding, (h0, c0))  # [seq_len, batch_size, hidden_size*2]
        # output = output[-1]  # [batch_size, hidden_size*2] 只需要最后一个输出

        # fc_output = self.fc(output)
        # fc_output = self.fc(hidden_cat)
        return output,hidden
