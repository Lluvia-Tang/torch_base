# Kim CNN Model

import torch
from torch import nn
import torch.nn.functional as F

def kmax_pooling(x, dim, k):

    index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]

    return x.gather(dim, index)

class KimCNN(nn.Module):

    def __init__(self, input_size, embedding_dim, n_filters, kernel_size, weights=None, net_dropout=0.5):

        super(KimCNN, self).__init__()
        self.in_channels = 1
        self.out_channels = 3
        self.model_name = 'Kim_CNN'
        
        self.dropout = nn.Dropout(net_dropout)
        if weights is not None:
            self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_dim, _weight=weights)
        else:
            self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_dim)
        self.k = 1
        # self.linear = nn.Linear(self.out_channels*4, input_size)
        self.linear = nn.Linear(len(kernel_size)* n_filters, input_size)

        self.fc = nn.Linear(len(kernel_size)* n_filters, 3)
        self.out = nn.Linear(input_size, 3) #3分类
        self.relu = nn.ReLU()
        # print(input_size)

        # print("+++++input_size:+++++\n", input_size)
        # self.convs = nn.ModuleList([nn.Conv2d(in_channels=1,
        #                                       out_channels=out_channels,kernel_size=K)
        #                             for K in kernel_size])

        #conv: [input_channel(=1), output_channel, (filter_height, filter_width), stride=1]
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1,
                                              out_channels=n_filters, kernel_size=(K, embedding_dim))
                                    for K in kernel_size])

    def forward(self, x):
        #x=[batch_size, seq_len]
        x = x[0]
        batch_size, seq_len = x.shape

        embedded = self.embedding(x)
        #embedded = [batchsize, seq_len, embedding_dim]

        embedded = embedded.unsqueeze(1)
        # print(embedded.shape)
        #embedded = [batchsize, (in_channel)1, seq_len, embedding_dim]

        conved = [self.relu(conv(embedded).squeeze(3)) for conv in self.convs]
        #conved = [batchsize, n_filters*out_channels, text len-filter_size[n] +1]

        # pooled = [kmax_pooling(i, 2, self.k).view(-1,out_channels*self.k) for i in conved]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        #pooled = [batchsize, n_filters*outchannels]

        # print(pooled)
        # print(len(pooled))
        # cat = torch.cat(pooled, dim=1)
        cat = self.dropout(torch.cat(pooled, dim=1))
        #cat = [batchsize, n_filters * len(filter_sizes)]
        # print("cat.shape: \n",cat.shape)
        linear = self.relu(self.linear(cat))
        linear = self.dropout(linear)
        # out = self.fc(linear)
        out = self.out(linear)
        # out = self.fc(cat)
        
        return out
