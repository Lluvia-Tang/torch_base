# ATGRU
# Paper "Connecting targets to tweets: Semantic attention-based model for target-specific stance detection"
# http://dro.dur.ac.uk/25714/1/25714.pdf

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

# def Attention_Stance(hidden_unit, W_h, W_z, b_tanh, v, length, target_word):
#
#     s1 = hidden_unit.size(0)
#     s2 = hidden_unit.size(1)
#     s3 = hidden_unit.size(2)
#
#     word_tensor = torch.zeros(s1,s2,1024).cuda()
#     word_tensor[:,:,:] = target_word
#
#     m1 = torch.mm(hidden_unit.view(-1,hidden_unit.size(2)),W_h).view(-1, s2, s3)
#     m2 = torch.mm(word_tensor.view(-1,1024),W_z).view(-1, s2, s3)
#     sum_tanh = F.tanh(m1 + m2 + b_tanh.unsqueeze(0))
#     u = torch.mm(sum_tanh.view(-1,s3),v.unsqueeze(1)).view(-1,s2,1).squeeze(2)
#
#     for i in range(len(length)):
#         u[i,length[i]:] = torch.Tensor([-1e6])
#     alphas = nn.functional.softmax(u)
#
#     context = torch.bmm(alphas.unsqueeze(1), hidden_unit).squeeze(1)
#
#     return context, alphas


class ATGRU(nn.Module):

    def __init__(self, linear_size, lstm_hidden_size,embedding_dim, embedding_matrix, net_dropout):

        super(ATGRU, self).__init__()

        self.hidden_dim = lstm_hidden_size
        self.embedding_dim = embedding_dim


        # self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # self.word_embeddings.weight = nn.Parameter(torch.tensor(embedding_matrix,dtype=torch.float))
        # self.word_embeddings.weight.requires_grad=True
        self.word_embeddings = nn.Embedding(num_embeddings=linear_size, embedding_dim=embedding_dim,
                                            _weight=embedding_matrix)

        self.attention = nn.Linear(2 * embedding_dim, 1)

        self.gru = nn.GRU(embedding_dim, self.hidden_dim, dropout=0.2, bidirectional=True,batch_first=True)

        self.dropout = nn.Dropout(net_dropout)

        self.hidden2target = nn.Linear(2 * self.hidden_dim, 3)
        # else:
        #     self.hidden2target = nn.Linear(self.hidden_dim, n_targets)

        # self.hidden = self.init_hidden()
        self.linear = nn.Linear(self.hidden_dim * 2, linear_size)
        self.out = nn.Linear(linear_size, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        text_indices, target = x[0],x[2]
        batch_size, seq_len = text_indices.shape

        h0 = torch.randn(1 * 2, batch_size, self.hidden_dim)
        h0 = h0.to(device="cuda:0")
        # c0 = torch.randn(1 * 2, batch_size, self.hidden_dim)
        # c0 = c0.to(device="cuda:0")

        x_emb = self.word_embeddings(text_indices)  # ([batchsize, 93, embedding_dim])

        t_emb = self.word_embeddings(target)
        # print(t_emb.shape)
        t_emb = torch.mean(t_emb, dim=1, keepdim=True)  # ([batchsize, 1, embedding_dim])

        # print("x_emb.shape :",x_emb.shape)

        xt_emb = torch.cat((x_emb, t_emb.expand(batch_size, seq_len, self.embedding_dim)), dim=2)  # ([100,93,600])

        a = self.attention(xt_emb)  # ([100, 93, 1])
        a = torch.transpose(a, 2, 1)

        # lstm_out, _ = self.lstm(x_emb.view(seq_len, 1 , self.embedding_dim))
        lstm_out, _ = self.gru(x_emb, h0)  # ([100, 93, 1024])

        # final_hidden_state = torch.mm(F.softmax(a.view(1,-1),dim=1),lstm_out.view(seq_len,-1))
        final_hidden_state = torch.bmm(a, lstm_out).squeeze(1)

        # target_space = self.hidden2target(self.dropout(final_hidden_state))
        # target_scores = F.log_softmax(target_space, dim=1)
        linear = self.relu(self.linear(final_hidden_state))
        linear = self.dropout(linear)
        out = self.out(linear)

        return out

    #     self.model_name = 'ATGRU'
    #
    #     self.dropout = nn.Dropout(net_dropout)
    #     self.embedding_dim = embedding_dim
    #     self.word_embeddings = nn.Embedding(num_embeddings=linear_size, embedding_dim=embedding_dim,
    #                                         _weight=embedding_matrix)
    #     self.hidden_dim = lstm_hidden_size
    #     self.gru = nn.GRU(embedding_dim, self.hidden_dim, dropout=lstm_dropout, bidirectional=True,batch_first=True)
    #     self.linear = nn.Linear(self.hidden_dim*2, linear_size)
    #     self.out = nn.Linear(linear_size, 3)
    #     self.relu = nn.ReLU()
    #     self.attention = nn.Linear(2 * embedding_dim, 1)
    #
    #
    #     # self.W_h = nn.Parameter(torch.rand([self.hidden_size*2,self.hidden_size*2],requires_grad=True))
    #     # self.W_z = nn.Parameter(torch.rand([self.hidden_size*2,self.hidden_size*2],requires_grad=True))
    #     # self.b_tanh = nn.Parameter(torch.rand(self.hidden_size*2,requires_grad=True))
    #     # self.v = nn.Parameter(torch.rand(self.hidden_size*2,requires_grad=True))
    #
    # def forward(self, x, target):
    #     batch_size, x_len = x.shape
    #
    #     a = torch.zeros(batch_size, 4)
    #     a = a.long()
    #     for i in range(batch_size):
    #         a[i, :] = target
    #     # target = a.t()
    #     target = a.to("cuda:0")
    #     h0 = torch.randn(1 * 2, batch_size, self.hidden_dim)
    #     h0 = h0.to(device="cuda:0")
    #     c0 = torch.randn(1 * 2, batch_size, self.hidden_dim)
    #     c0 = c0.to(device="cuda:0")
    #
    #     x_emb = self.word_embeddings(x)  # ([batchsize, 93, embedding_dim])
    #
    #     t_emb = self.word_embeddings(target)
    #     # print(t_emb.shape)
    #     t_emb = torch.mean(t_emb, dim=1, keepdim=True)  # ([batchsize, 1, embedding_dim])
    #
    #     # print("x_emb.shape :",x_emb.shape)
    #
    #     xt_emb = torch.cat((x_emb, t_emb.expand(batch_size, x_len, self.embedding_dim)), dim=2)  # ([100,93,600])
    #
    #     a = self.attention(xt_emb)  # ([100, 93, 1])
    #     a = torch.transpose(a, 2, 1)
    #
    #     # lstm_out, _ = self.lstm(x_emb.view(seq_len, 1 , self.embedding_dim))
    #     gru_out, (_, _) = self.gru(x_emb, (h0, c0))  # ([100, 93, 1024])
    #
    #     # final_hidden_state = torch.mm(F.softmax(a.view(1,-1),dim=1),lstm_out.view(seq_len,-1))
    #     final_hidden_state = torch.bmm(a, gru_out).squeeze(1)
    #
    #     # packed_output, (ht, ct) = self.lstm(packed_input)
    #     # output, _ = pad_packed_sequence(packed_output,batch_first=True)
    #
    #
    #     atten = self.dropout(final_hidden_state)
    #
    #     linear = self.relu(self.linear(atten))
    #     linear = self.dropout(linear)
    #     out = self.out(linear)
    #
    #     return out
