# -*- coding:utf-8 -*-
__author__ = 'tangs'
__time__ = '2021/12/3 19:30'

from torch import nn
from transformers import BertModel

class bertLSTM(nn.Module):
    def __init__(self,model,num_classes):
        super(bertLSTM,self).__init__()
        self.bert=BertModel.from_pretrained(model)  #从路径加载预训练模型
        for param in self.bert.parameters():
            param.requires_grad = True # 使参数可更新
        self.hidden_size = 1024
        self.rnn_hidden_size = 512
        self.num_layers = 1
        self.dropout = 0.5

        # batch_first 参数设置batch维度在前还是序列维度在前（输入输出同步改变）
        # 在每个RNN神经元之间使用 dropout 在单个神经元内部的两个时间步长间不使用 dropout
        self.lstm=nn.LSTM(self.hidden_size,self.rnn_hidden_size,self.num_layers,batch_first=True,dropout=self.dropout,bias=True,bidirectional=True)
        self.dropout=nn.Dropout(self.dropout)

        # 双向LSTM要*2 分析LSTM节点数和网络层数时，看成神经元是LSTM全连接网络
        self.fc=nn.Linear(self.rnn_hidden_size*2,num_classes) # 自定义全连接层 ，输入数（输入的最后一个维度），输出数（多分类数量），bert模型输出的最后一个维度是768，这里的输入要和bert最后的输出统一

    def forward(self,x):
        context, mask = x

        # 第一个参数 是所有输入对应的输出  第二个参数 是 cls最后接的分类层的输出
        outputs = self.bert(context,attention_mask=mask) # output_all_encoded_layers 是否将bert中每层(12层)的都输出，false只输出最后一层【128*768】
        output =self.lstm(outputs.last_hidden_state) # 128*10
        # out = self.lstm(encoder)  # encoder维度[batch_size,pad_size,bert_hidden_size]，out的维度为 [batch_size,pad_size,lstm_hidden_size]
        out = self.dropout(output)
        out = out[:, -1, :]  # 只要序列中最后一个token对应的输出，（因为lstm会记录前边token的信息）
        out = self.fc(out)
        return out
