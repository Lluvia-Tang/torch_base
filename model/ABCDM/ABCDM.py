# -*- coding:utf-8 -*-
__author__ = 'tangs'
__time__ = '2021/12/17 10:20'
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.AT.selfAttention import Attention
'''
ABCDM 
input ->BiGRU  ->attention  -> CNN  -> concat -> out
      ->BiLSTM ->attention  -> CNN  ->
'''

class ABCDM(nn.Module):

    def __init__(self,input_size, embedding_dim, hidden_size, n_filters, kernel_size, weights=None, dropout=0.5):
        super(ABCDM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.bilstm = BiLSTM(input_size, embedding_dim, hidden_size, weights)
        self.bi_gru = BiGRU(input_size, hidden_size,embedding_dim, weights)
        self.attention = Attention(2 * hidden_size, 2 * hidden_size, 2 * hidden_size)
        self.cnn = TextCNN(input_size, embedding_dim, n_filters, kernel_size, weights)
        self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_dim, _weight=weights)
        self.out = nn.Linear(input_size * 2 , 3)  # 3分类

    def forward(self, x):
        text = x[0]
        batch_size, seq_len = text.shape
        embeded = self.embedding(text)  #[batch, seq_len, emdedding_dim]
        # print("all ABCDM batch : ", batch_size)
        # print("送入编码器shape : ",embeded.shape)

        lstm_out, lstm_hidden = self.bilstm(embeded) # hidden [n_layers * 2, batch_size, hidden_size] out [seq, batch, hidden *2]
        gru_out, gru_hidden = self.bi_gru(embeded)  #hidden [n_layers * 2, batch_size, hidden_size]
        if isinstance(lstm_hidden, tuple):  # LSTM
            lstm_hidden = lstm_hidden[0]  # take the final_hidden_state
        lstm_hidden = torch.cat([lstm_hidden[-1], lstm_hidden[-2]], dim=1) #need to concat the last 2 hidden layers (bi-)
        gru_hidden = torch.cat([gru_hidden[-1], gru_hidden[-2]], dim=1)

        #attention Outputs = a:[seq x batch], lin_comb:[batch x hidden*2]
        lstm_energy, lstm_atten = self.attention(lstm_hidden, lstm_out, lstm_out)
        gru_energy, gru_atten = self.attention(gru_hidden, gru_out, gru_out)

        #cnn
        lstm_linear = self.cnn(lstm_atten) #[batch, inputsize]
        gru_linear = self.cnn(gru_atten)  #[batch, inputsize]

        linear = torch.cat((lstm_linear, gru_linear), dim=1) #[batch, inputsize*2]
        out = self.dropout(linear)
        out = self.out(out)

        return out


class BiLSTM(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, weights=None, dropout=0.5):

        super(BiLSTM, self).__init__()

        self.model_name = 'BiLSTM'
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.n_layers = 1

        self.lstm = nn.LSTM(embedding_dim, hidden_size, dropout=dropout, bidirectional=True)
        self.fc = torch.nn.Linear(hidden_size * 2, 3)  # 分成3类

    def forward(self, input):
        # input shape : B x S x embedding
        batch_size, _,_ = input.shape
        input = input.transpose(0,1)  #[seqlen, batch, embedding]
        h0 = torch.randn(self.n_layers * 2, batch_size, self.hidden_size)
        h0 = h0.to(device="cuda:0")
        c0 = torch.randn(self.n_layers * 2, batch_size, self.hidden_size)
        c0 = c0.to(device="cuda:0")

        output, hidden = self.lstm(input, (h0, c0))  # [seq_len, batch_size, hidden_size*2]
        # (final_hidden_state, final_cell_state)
        # output = output[-1]  # [batch_size, hidden_size*2] 只需要最后一个输出
        # hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        # fc_output = self.fc(hidden_cat)
        return output, hidden

class BiGRU(nn.Module):

    def __init__(self, linear_size, lstm_hidden_size,embedding_dim, embedding_matrix, net_dropout=0.5):
        super(BiGRU, self).__init__()

        self.hidden_dim = lstm_hidden_size
        self.embedding_dim = embedding_dim
        self.word_embeddings = nn.Embedding(num_embeddings=linear_size, embedding_dim=embedding_dim,
                                            _weight=embedding_matrix)

        self.gru = nn.GRU(embedding_dim, self.hidden_dim, dropout=0.2, bidirectional=True)

        self.dropout = nn.Dropout(net_dropout)
        self.hidden2target = nn.Linear(2 * self.hidden_dim, 3)

        self.linear = nn.Linear(self.hidden_dim * 2, linear_size)
        self.out = nn.Linear(linear_size, 3)
        self.relu = nn.ReLU()

    def forward(self, x_emb):
        batch_size, _,_ = x_emb.shape
        x_emb = x_emb.transpose(0,1) #[seq_len, batch_size, embedding]

        h0 = torch.randn(1 * 2, batch_size, self.hidden_dim)
        h0 = h0.to(device="cuda:0")
        # print("BiGRU shape : ",x_emb.shape)
        # lstm_out, _ = self.lstm(x_emb.view(seq_len, 1 , self.embedding_dim))
        gru_out, hidden = self.gru(x_emb, h0)

        return gru_out, hidden


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index)

class TextCNN(nn.Module):
    def __init__(self, input_size, embedding_dim, n_filters, kernel_size, weights=None, net_dropout=0.5):

        super(TextCNN, self).__init__()
        self.in_channels = 1
        self.out_channels = 1
        self.model_name = 'Kim_CNN'
        self.dropout = nn.Dropout(net_dropout)

        self.linear = nn.Linear(len(kernel_size) * n_filters, input_size)

        self.fc = nn.Linear(len(kernel_size) * n_filters, 3)
        # self.out = nn.Linear(input_size, 3)  # 3分类
        self.relu = nn.ReLU()

        # conv: [input_channel(=1), output_channel, (filter_height, filter_width), stride=1]
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1,
                                              out_channels=n_filters, kernel_size=(K, 1))
                                    for K in kernel_size])

    def forward(self, x):

        embedded = x.unsqueeze(1)
        embedded = embedded.unsqueeze(3)
        # embedded = [batchsize, (in_channel)1, hidden*2, 1]

        conved = [self.relu(conv(embedded).squeeze(3)) for conv in self.convs]
        # conved = [batchsize, n_filters*out_channels, seq_len - filter_size[n] +1]  [([64, 32, 1023])]
        # print(conved[0].shape)
        # pooled = [kmax_pooling(i, 2, self.k).view(-1,out_channels*self.k) for i in conved]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled = [batchsize, n_filters*outchannels]

        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = [batchsize, n_filters * len(filter_sizes)]

        linear = self.relu(self.linear(cat))   #[batchsize, inputsize]
        # linear = self.dropout(linear)

        # out = self.out(linear)

        return linear