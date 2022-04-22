#-*- codeing = utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import functional as F
import math



class textCNN(nn.Module):
    def __init__(self,input_size, embedding_dim, weights=None, net_dropout=0.5):
        super(textCNN, self).__init__()
        n_filters=16
        filter_sizes=[3,4,5]
        self.dropout = nn.Dropout(net_dropout)

        if weights is not None:
            self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_dim, _weight=weights)
        else:
            self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_dim)

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(n_filters * len(filter_sizes), 3)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        embedded = self.embedding(inputs)
        x = embedded.unsqueeze(1)
        convd = [self.relu(conv(x).squeeze(3)) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in convd]
        cat = self.dropout(torch.cat(pooled, 1))
        return self.fc(cat)
'''
        self.opt = opt
        ci = 1 #input chanel size
        co = 3
        kernel_size = [2,3,4]
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))

        self.conv11 = nn.Conv2d(ci, co, (kernel_size[0], opt.train['embed_dim'])),#(filter height,filter width)

        # self.conv11 = nn.Sequential(
        #     nn.Conv2d(ci, co, (kernel_size[0], opt.train['embed_dim'])),#(filter height,filter width)
        #     nn.ReLU(),
        #     nn.MaxPool2d((2,1)) #(filter height,filter width)
        # )
        # self.conv12 = nn.Sequential(
        #     nn.Conv2d(ci, co, (kernel_size[1], opt.train['embed_dim'])),  # (filter height,filter width)
        #     nn.ReLU(),
        #     nn.MaxPool2d((2, 1))  # (filter height,filter width)
        # )
        # self.conv13 = nn.Sequential(
        #     nn.Conv2d(ci, co, (kernel_size[2], opt.train['embed_dim'])),  # (filter height,filter width)
        #     nn.ReLU(),
        #     nn.MaxPool2d((2, 1))  # (filter height,filter width)
        # )

        self.conv12 = nn.Conv2d(ci, co, (kernel_size[1], opt.train['embed_dim']))
        self.conv13 = nn.Conv2d(ci, co, (kernel_size[2], opt.train['embed_dim']))
        #self.text_embed_dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(co*len(kernel_size), opt.tasks_num_class[0])

    @staticmethod
    def conv_and_pool(x, conv):
        # x: (batch, 1, sentence_length,  )
        x = conv(x)
        # x: (batch, kernel_num, H_out, 1)
        x = F.relu(x.squeeze(3))
        # x: (batch, kernel_num, H_out)
        x = F.max_pool1d(x, x.shape[2]).squeeze(2)
        #  (batch, kernel_num)
        return x

    def forward(self, inputs):
        text_indices = inputs[0]
        x = self.embed(text_indices)
        x = x.unsqueeze(1)
        # x1 = self.conv11(x)
        # x2 = self.conv12(x)
        # x3 = self.conv13(x)
        # x = torch.cat((x1, x2, x3), 1)
        x1 = self.conv_and_pool(x,self.conv11)
        x2 = self.conv_and_pool(x, self.conv12)
        x3 = self.conv_and_pool(x, self.conv13)
        x = torch.cat((x1, x2, x3), 1)
        #flatten = conved.view(self.opt.train['batch_size'],-1) #batchsize,output_channel
        output = F.log_softmax(self.fc(x), dim=1)

        #x = F.softmax(x.sum(1, keepdim=True), dim=2).squeeze(1)
        #output = self.stance_classifier(x)
        # output = self.fc(x)

        return output
'''

