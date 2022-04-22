# TAN
# Paper "Stance Classification with Target-Specific Neural Attention Networks"
# https://www.ijcai.org/Proceedings/2017/0557.pdf

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

class LSTM_TAN(nn.Module):
    def __init__(self,embedding_dim, hidden_dim, vocab_size, n_targets,embedding_matrix,dropout = 0.5):
        super(LSTM_TAN, self).__init__()
        version = "tan"

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # self.word_embeddings.weight = nn.Parameter(torch.tensor(embedding_matrix,dtype=torch.float))
        # self.word_embeddings.weight.requires_grad=True
        self.word_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, _weight=embedding_matrix)
        self.version = version

        self.attention = nn.Linear(2*embedding_dim,1)


        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)

        self.dropout = nn.Dropout(dropout)

        self.hidden2target = nn.Linear(2*self.hidden_dim, n_targets)
        # else:
        #     self.hidden2target = nn.Linear(self.hidden_dim, n_targets)

        # self.hidden = self.init_hidden()
        self.linear = nn.Linear(self.hidden_dim * 2, vocab_size)
        self.out = nn.Linear(vocab_size, 3)
        self.relu = nn.ReLU()


    def forward(self, x):
        text_indices, target = x[0],x[2]
        batch_size, seq_len = text_indices.shape

        h0 = torch.randn(1 * 2, batch_size, self.hidden_dim)
        h0 = h0.to(device="cuda:0")
        c0 = torch.randn(1 * 2, batch_size, self.hidden_dim)
        c0 = c0.to(device="cuda:0")


        x_emb = self.word_embeddings(text_indices)#([batchsize, 93, embedding_dim])

        t_emb = self.word_embeddings(target)
        # print(t_emb.shape)
        t_emb = torch.mean(t_emb,dim=1,keepdim=True) #([batchsize, 1, embedding_dim])

        # print("x_emb.shape :",x_emb.shape)

        xt_emb = torch.cat((x_emb,t_emb.expand(batch_size,seq_len,self.embedding_dim)),dim=2) #([100,93,600])

        a = self.attention(xt_emb) #([100, 93, 1])
        a = torch.transpose(a,2,1)

        # lstm_out, _ = self.lstm(x_emb.view(seq_len, 1 , self.embedding_dim))
        lstm_out, (_, _) = self.lstm(x_emb, (h0, c0)) #([100, 93, 1024])

        # final_hidden_state = torch.mm(F.softmax(a.view(1,-1),dim=1),lstm_out.view(seq_len,-1))
        final_hidden_state = torch.bmm(a, lstm_out).squeeze(1)

        # target_space = self.hidden2target(self.dropout(final_hidden_state))
        # target_scores = F.log_softmax(target_space, dim=1)
        linear = self.relu(self.linear(final_hidden_state))
        linear = self.dropout(linear)
        out = self.out(linear)

        return out

        # return target_scores



# def Attention_Stance(hidden_unit, h_embedding2, W_h, b_tanh, length, target_word):
#
#     word_tensor = torch.zeros(h_embedding2.size(0),h_embedding2.size(1),1024).cuda()
#     word_tensor[:,:,:] = target_word
#     word_tensor = torch.cat((h_embedding2,word_tensor),2)
#
#     s1 = h_embedding2.size(0)
#     s2 = h_embedding2.size(1)
#
#     m1 = torch.mm(word_tensor.view(-1,2048),W_h).view(s1, s2, -1)
#     u = (m1 + b_tanh.unsqueeze(1)).squeeze(2)
#
#     for i in range(len(length)):
#         u[i,length[i]:] = torch.Tensor([-1e6])
#
#     alphas = nn.functional.softmax(u)
#
#     context = torch.bmm(alphas[:,:hidden_unit.size(1)].unsqueeze(1), hidden_unit).squeeze(1)
#
#     return context, alphas
#
#
# class TAN(nn.Module):
#
#     def __init__(self, sequence_length, linear_size, lstm_hidden_size, net_dropout, lstm_dropout):
#
#         super(TAN, self).__init__()
#
#         self.model_name = 'TAN'
#
#         self.dropout = nn.Dropout(net_dropout)
#
#         self.hidden_size = lstm_hidden_size
#         self.lstm = nn.LSTM(1024, self.hidden_size, dropout=lstm_dropout, bidirectional=True)
#         self.linear = nn.Linear(self.hidden_size*2, linear_size)
#         self.out = nn.Linear(linear_size, 3)
#         self.relu = nn.ReLU()
#
#         self.W_h = nn.Parameter(torch.rand([2*1024,1],requires_grad=True))
#         self.b_tanh = nn.Parameter(torch.rand(sequence_length,requires_grad=True))
#
#     def forward(self, x, x_len, epoch, target_word, tokens):
#         x = x.squeeze(1)
#         target_word = target_word.squeeze(1)
#         if target_word.size(1) != 1024:
#             target_word = target_word.sum(1) / target_word.size(1)
#         target_word = target_word.unsqueeze(1)
#
#         seq_lengths, perm_idx = x_len.sort(0, descending=True)
#         seq_tensor = x[perm_idx,:,:]
#         packed_input = pack_padded_sequence(seq_tensor, seq_lengths, batch_first=True)
#         packed_output, (ht, ct) = self.lstm(packed_input)
#         output, _ = pad_packed_sequence(packed_output,batch_first=True)
#         _, unperm_idx = perm_idx.sort(0)
#         h_lstm = output[unperm_idx,:,:]
#
#         atten, alpha = Attention_Stance(h_lstm,x,self.W_h,self.b_tanh,x_len,epoch,tokens, target_word)
#         atten = self.dropout(atten)
#
#         linear = self.relu(self.linear(atten))
#         linear = self.dropout(linear)
#         out = self.out(linear)
#
#         return out
