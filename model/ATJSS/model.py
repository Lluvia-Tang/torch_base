# 3rd-party module
import torch
from torch import nn
from torch.nn import functional as F

class BaseModel(torch.nn.Module):
    def __init__(self, linear_size, hidden_size, embedding_dim, embedding_matrix, dropout):
        super(BaseModel, self).__init__()

        # config
        self.embedding_dim = embedding_dim
        self.rnn_dropout = 0.2
        self.hidden_dim = hidden_size
        self.linear_size = linear_size
        self.stance_linear_dim = 100
        self.sentiment_linear_dim = 50
        self.linear_dropout = dropout
        self.output_dim = 3 #分类数

        # embedding layer
        self.embedding_layer = nn.Embedding(num_embeddings=linear_size, embedding_dim=embedding_dim,
                                            _weight=embedding_matrix)

        # attention layer for stance
        self.stance_attn = StanceAttn(self.embedding_dim, self.hidden_dim, self.rnn_dropout)

        # attention layer for sentiment
        self.sentiment_attn = SentimentAttn(self.embedding_dim, self.hidden_dim, self.rnn_dropout)

        # linear layer for stance
        self.stance_linear = StanceLinear(self.hidden_dim, self.stance_linear_dim,self.linear_dropout, self.output_dim)

        # linear layer for sentiment
        self.sentiment_linear = SentimentLinear(self.hidden_dim, self.sentiment_linear_dim,self.linear_dropout, self.output_dim)

        self.rnn_dp = nn.Dropout(p=self.rnn_dropout)

    def forward(self, x):#batch_target, batch_claim
        batch_claim, target = x[0], x[2]
        # print(batch_claim.shape)
        # get target embedding
        # get average of target embedding
        t_emb = self.embedding_layer(target)
        # print(t_emb.shape)
        batch_target = torch.mean(t_emb, dim=1, keepdim=True)  # ([batchsize, 1, embedding_dim])

        # embedding layer
        batch_claim = self.rnn_dp(self.embedding_layer(batch_claim))

        # stance attn
        stance_r, stance_weight = self.stance_attn(batch_target, batch_claim)

        # sentiment attn
        sentiment_r, sentiment_weight = self.sentiment_attn(batch_claim)

        # stance linear and softmax
        stance_r = self.stance_linear(stance_r, sentiment_r)

        # sentiment linear and softmax
        sentiment_r = self.sentiment_linear(sentiment_r)

        return stance_r, sentiment_r, stance_weight, sentiment_weight

class StanceAttn(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, rnn_dropout):
        super(StanceAttn, self).__init__()

        # config
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_rnn_layers = 2
        self.rnn_dropout = rnn_dropout


        # get parameters of LSTM

        parameter = {'input_size': self.embedding_dim,
                     'hidden_size': self.hidden_dim,
                     'num_layers': self.num_rnn_layers,
                     'batch_first': True,
                     'dropout': self.rnn_dropout,
                     'bidirectional': True}


        # bidirectional LSTM
        self.LSTM = nn.LSTM(**parameter)

        # linear layer for W_t t + b_t
        self.t_linear = nn.Linear(in_features=self.embedding_dim,
                                  out_features=2*self.hidden_dim,
                                  bias=True)

        # linear layer for W_i' h_i
        self.h_linear = nn.Linear(in_features=2*self.hidden_dim,
                                  out_features=2*self.hidden_dim,
                                  bias=False)

        # linear layer for v_s^T
        self.v_linear = nn.Linear(in_features=2*self.hidden_dim,
                                  out_features=1,
                                  bias=False)

    def forward(self, batch_target, batch_claim):
        # batch_size = batch_claim[0]  #[batchsize, 93, embedding_dim]
        # seq_len = batch_claim[1]
        # get all hidden vector
        claim_ht, _ = self.LSTM(batch_claim)  # (B, S, H)
        #target [batch_Size,1,embedding_dim]
        # get attention vector e
        e = torch.tanh(self.t_linear(batch_target) + # (B, 1, H)
                       self.h_linear(claim_ht))  # (B, S, H)

        e = self.v_linear(e).squeeze(dim=2)  # (B, S)

        # apply softmax to get attention score
        weight = torch.nn.functional.softmax(e, dim=1)  # (B, S)

        # print(weight.unsqueeze(1).shape)
        # print(claim_ht.shape)
        # get final vector representation
        r = torch.matmul(weight.unsqueeze(1), claim_ht).squeeze(1)  # (B, H)

        return r, weight

class SentimentAttn(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, rnn_dropout):
        super(SentimentAttn, self).__init__()

        # config
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_rnn_layers = 2
        self.rnn_dropout = rnn_dropout

        # get parameters of LSTM

        parameter = {'input_size': self.embedding_dim,
                     'hidden_size': self.hidden_dim,
                     'num_layers': self.num_rnn_layers,
                     'batch_first': True,
                     'dropout': self.rnn_dropout,
                     'bidirectional': True}

        # bidirectional LSTM
        self.LSTM = nn.LSTM(**parameter)

        # linear layer for W_s s + b_s
        self.s_linear = nn.Linear(in_features=2*self.hidden_dim,
                                  out_features=2*self.hidden_dim,
                                  bias=True)

        # linear layer for W_i h_i
        self.h_linear = nn.Linear(in_features=2*self.hidden_dim,
                                  out_features=2*self.hidden_dim,
                                  bias=False)

        # linear layer for v_t^T
        self.v_linear = nn.Linear(in_features=2*self.hidden_dim,
                                  out_features=1,
                                  bias=False)

    def forward(self, batch_claim):
        # get all hidden vector
        claim_ht, _ = self.LSTM(batch_claim)  # (B, S, H)

        # get final hidden vector
        final_claim_ht = claim_ht[:, -1]  # (B, H)

        # get attention vector e
        e = torch.tanh(self.s_linear(final_claim_ht).unsqueeze(1) + # (B, 1, H)
                       self.h_linear(claim_ht))  # (B, S, H)

        e = self.v_linear(e).squeeze(dim=2)  # (B, S)

        # apply softmax to get attention score
        weight = torch.nn.functional.softmax(e, dim=1)  # (B, S)

        # get final vector representation
        r = torch.matmul(weight.unsqueeze(1), claim_ht).squeeze(1)  # (B, H)

        return r, weight

class StanceLinear(torch.nn.Module):
    def __init__(self, hidden_dim, stance_linear_dim,linear_dropout, output_dim):
        super(StanceLinear, self).__init__()

        self.hidden_dim = hidden_dim
        self.stance_linear_dim = stance_linear_dim
        self.linear_dropout = linear_dropout
        self.output_dim = output_dim

        self.linear = nn.Sequential(
            # nn.Dropout(p=config.linear_dropout),
            nn.Linear(in_features=4*self.hidden_dim,
                      out_features=self.stance_linear_dim),
            nn.ReLU(),
            nn.Dropout(p=self.linear_dropout),
            nn.Linear(in_features=self.stance_linear_dim,
                      out_features=self.output_dim),
        )

    def forward(self, stance_r, sentiment_r):
        stance_r = torch.cat((sentiment_r, stance_r), dim=1)
        stance_r = self.linear(stance_r)

        return stance_r

class SentimentLinear(torch.nn.Module):
    def __init__(self, hidden_dim, sentiment_linear_dim,linear_dropout, output_dim):
        super(SentimentLinear, self).__init__()
        self.hidden_dim = hidden_dim
        self.sentiment_linear_dim = sentiment_linear_dim
        self.linear_dropout = linear_dropout
        self.output_dim = output_dim

        self.linear = nn.Sequential(
            # nn.Dropout(p=config.linear_dropout),
            nn.Linear(in_features=2*self.hidden_dim,
                        out_features=self.sentiment_linear_dim),
            nn.ReLU(),
            nn.Dropout(p=self.linear_dropout),
            nn.Linear(in_features=self.sentiment_linear_dim,
                        out_features=self.output_dim),
        )

    def forward(self, sentiment_r):
        sentiment_r = self.linear(sentiment_r)

        return sentiment_r