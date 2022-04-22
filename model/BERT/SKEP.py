# -*- coding:utf-8 -*-
__author__ = 'tangs'
__time__ = '2021/12/18 12:39'
import re
import math
import torch
import numpy as np
from random import *
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from transformers import BertPreTrainedModel, BertModel
from paddlenlp.transformers import SkepForSequenceClassification, SkepTokenizer

class SkepForClassification(nn.Module):
	def __init__(self, model,num_classes):
		super(SkepForClassification,self).__init__()
		self.bert = SkepForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model, num_classes=3)
		for param in self.bert.parameters():
			param.requires_grad = True  # 使参数可更新
		# self.hidden_size = 1024  #covid-bert的embedding是1024不同于768
		self.hidden_size = 768
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
		out = self.fc(outputs.pooler_output)  # batchsize*3
		# print("out:",out)
		return out

