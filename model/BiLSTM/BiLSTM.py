# # BiLSTM

import torch
from torch import nn


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
            self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_dim,_weight=weights)
        else:
            self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_dim)
        # self.embedding = torch.nn.Embedding(input_size, hidden_size)

        self.lstm = nn.LSTM(embedding_dim, hidden_size, dropout=dropout, bidirectional=True)
        self.fc = torch.nn.Linear(hidden_size * 2, 3) #分成3类
        # self.linear = nn.Linear(self.hidden_size*2, linear_size)
        # self.out = nn.Linear(linear_size, 3)
        # self.relu = nn.ReLU()
        
    def forward(self, input):

        # input shape : B x S -> S x B
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

        embedding = self.embedding(input)

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
        output, (_, _) = self.lstm(embedding, (h0,c0))  #[seq_len, batch_size, hidden_size*2]
        output = output[-1]  #[batch_size, hidden_size*2] 只需要最后一个输出
        # hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        fc_output = self.fc(output)
        # fc_output = self.fc(hidden_cat)
        return fc_output
