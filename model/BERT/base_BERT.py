# -*- coding:utf-8 -*-
__author__ = 'tangs'
__time__ = '2021/12/1 19:07'
import re
import math
import torch
import numpy as np
from random import *
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from transformers import BertPreTrainedModel, BertModel


class BertForClassification(nn.Module):
	def __init__(self, model,num_classes):
		super(BertForClassification,self).__init__()
		self.bert = BertModel.from_pretrained(model)
		# self.bert = BertModel.from_pretrained(model, output_attentions=True)   #可视化attention
		for param in self.bert.parameters():
			param.requires_grad = True  # 使参数可更新
		self.hidden_size = 1024  #covid-bert的embedding是1024不同于768
		# self.hidden_size = 768
		# self.dropout = nn.Dropout(config.hidden_dropout_prob)
		#The classification layer that takes the [CLS] representation and outputs the logit
		# self.hidden_layer = nn.Linear(model.hidden_size, model.hidden_size)
		self.fc = nn.Linear(self.hidden_size, num_classes)

	def forward(self, input):
		'''
		Inputs:
			-input_ids : Tensor of shape [B, T] containing token ids of sequences
			-attention_mask : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
			(where B is the batch size and T is the input length)
		'''
		#Feed the input to Bert model to obtain outputs
		input_ids, attention_mask = input[0],input[1]
		# outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
		# 第一个参数 是所有输入对应的输出  第二个参数 是 cls最后接的分类层的输出
		outputs = self.bert(input_ids, attention_mask)  # output_all_encoded_layers 是否将bert中每层(12层)的都输出，false只输出最后一层【128*768】
		# print(outputs.pooler_output.shape)
		_,pool = outputs
		out = self.fc(pool)  # batchsize*3
		# print("out:",out)
		# attention = self.bert(input_ids)[-1]
		# return out, attention
		return out

'''
Bert输出
last_hidden_state：shape是(batch_size, sequence_length, hidden_size)，hidden_size=768,它是模型最后一层输出的隐藏状态。
					（通常用于命名实体识别）
pooler_output：shape是(batch_size, hidden_size)，这是序列的第一个token(classification token)的最后一层的隐藏状态，
			它是由线性层和Tanh激活函数进一步处理的。（通常用于句子分类，至于是使用这个表示，还是使用整个输入序列的隐藏状态序列的平均化或池化，视情况而定）

# attention.shape = num_layers * bs * num_head * seq_len * seq_len
attention = model(input_ids)[-1]
'''