# -*- coding:utf-8 -*-
__author__ = 'tangs'
__time__ = '2022/2/27 17:36'
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel
import random
from model import transformer

class DCRAN_BERT(nn.Module):
    def __init__(self, model, hidden_size):
        super(DCRAN_BERT, self).__init__()
        self.class_num = 3
        self.embedding_size_word = 768
        self.hidden_size = hidden_size
        self.num_hidden_layers = 2
        self.num_heads = 2
        self.attention_key_channels = 0
        self.attention_value_channels = 0
        self.filter_size = 64
        self.max_length = 800
        self.dropout = 0.2
        self.bert = BertModel.from_pretrained(model)  # 从路径加载预训练模型
        for param in self.bert.parameters():
            param.requires_grad = True  # 使参数可更新
        self.ffnn_A = nn.Linear(768,self.hidden_size) #AE前馈层
        self.ffnn_O = nn.Linear(768,self.hidden_size) #OE前馈层
        self.fc_A = nn.Linear(self.hidden_size, self.class_num)
        self.fc_O = nn.Linear(self.hidden_size, self.class_num)
        # Define Decoder
        self.decoder = transformer.Decoder(
            self.embedding_size_word,
            self.hidden_size,
            self.num_hidden_layers,
            self.num_heads,
            self.attention_key_channels,
            self.attention_value_channels,
            self.filter_size,
            self.max_length,
            self.dropout,
            self.dropout,
            self.dropout,
            self.dropout,
            use_mask=True
        )
        self.fc_SC = nn.Linear(self.hidden_size, self.class_num)
        self.fc_MT = nn.Linear(self.hidden_size, self.class_num)
        self.fc_R = nn.Linear(self.hidden_size, self.class_num)

    def forward(self,inputs):
        bert_input_ids, bert_input_mask, bert_segment_ids, word_mask, senti_mask, maskItem_ids, maskRel_ids = inputs
        #['text_indices','bert_mask','bert_segment','source_mask','sentiment_mask','maskItem_input','maskRel_input']
        # aspect_prob_list, opinion_prob_list, senti_prob_list = list()
        res = self.bert(input_ids = bert_input_ids,attention_mask = bert_input_mask,token_type_ids = bert_segment_ids)
        h = res[0] #(batch_size, seq_len, embed)[32,100,768]

        # h_cat = h[:,1:,:]
        # h_cat = h_cat[:,:-1,:] #h[1:n+1] 去除[CLS]和[SEP]  [32,98,768]

        z_a = self.ffnn_A(h)  #[32,100,768]
        z_o = self.ffnn_O(h)

        aspect_prob = self.fc_A(z_a)  #[32, 100, 3]
        opinion_prob = self.fc_O(z_o)

        word_mask = torch.unsqueeze(word_mask,-1)
        word_mask = word_mask.repeat(1,1,self.class_num)  #[32, 100, 3]

        aspect_prob = torch.reshape(word_mask * aspect_prob, [-1, self.class_num]) #[3200, 3]
        opinion_prob = torch.reshape(word_mask * opinion_prob, [-1, self.class_num])

        decoder_outputs, state = self.decoder((h, z_a, z_o))  # [batch, seq_len, embed]  [32, 100, 768]

        senti_prob = self.fc_SC(decoder_outputs)
        senti_mask = torch.unsqueeze(senti_mask, -1)
        senti_mask = senti_mask.repeat(1,1,self.class_num)
        senti_mask = senti_mask.type(torch.FloatTensor).to("cuda:1")

        sentiment_prob = torch.reshape(senti_mask * senti_prob, [-1, self.class_num]) #[3200, 3]

        #aux tasks
        #TSMTD
        res1 = self.bert(input_ids=maskItem_ids, attention_mask=bert_input_mask)
        tsmtd_prob = self.fc_MT(res1[1])

        #PRD
        res2 = self.bert(input_ids=maskRel_ids, attention_mask=bert_input_mask)
        prd_prob = self.fc_R(res2[1])

        return aspect_prob, opinion_prob, sentiment_prob, tsmtd_prob, prd_prob
