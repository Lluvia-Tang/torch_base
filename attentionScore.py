# -*- coding:utf-8 -*-
__author__ = 'tangs'
__time__ = '2022/2/15 14:00'
import numpy as np
import spacy
import torch
from transformers import BertTokenizer
from utils.clear import clean_str
import sys
np.set_printoptions(threshold=sys.maxsize)


# load model
from model.BERT.base_BERT import BertForClassification


def load_checkpoint(model, checkpoint_PATH, optimizer):
    model_CKPT = torch.load(checkpoint_PATH)
    model.load_state_dict(model_CKPT['state_dict'])
    print('loading checkpoint!')
    optimizer.load_state_dict(model_CKPT['optimizer'])
    return model, optimizer


# model = BertForClassification("kornosk/bert-political-election2020-twitter-mlm", 3)
# # model = TheModelClass()
# param_optimizer = list(model.named_parameters())
# # 不需要衰减的参数  #衰减：（修正的L2正则化）
# # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
# no_decay = ['bias', 'LayerNorm.weight']
# optimizmer_grouped_parameters = [
#             {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
#             # 遍历所有参数，如果参数名字里有no_decay的元素则取出元素
#             {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
#             # 遍历所有参数，如果参数名字里没有no_decay的元素则取出元素
#         ]
# optimizer = torch.optim.Adam(optimizmer_grouped_parameters, 1e-5,
#                                           betas=(0.9, 0.999),
#                                           weight_decay=4e-5)
#
# model, optimizer = load_checkpoint(model, "KMELM_model.tar", optimizer)
# model.eval()
model = torch.load('KMELM_model2.pth')
model.eval()
text = "I have next to no immune system right now so thanks to all wearing masks."
tokenizer = BertTokenizer.from_pretrained("kornosk/bert-political-election2020-twitter-mlm")
text = clean_str(text)
nlp = spacy.load('en_core_web_sm')
SEP,CLS='[SEP]','[CLS]'

doc = text.split(" ")
# doc = [str(token) for token in doc if not token.is_punct | token.is_stop]
token = [CLS] + doc + [SEP] #在序列前加一个[CLS]标志位
token_ids = tokenizer.convert_tokens_to_ids(token)
# print(token_ids)
attention_mask = [float(i > 0) for i in token_ids]

token_ids = np.array(token_ids)
token_ids = token_ids.reshape((1, 17))

attention_mask = np.array(attention_mask)
attention_mask = attention_mask.reshape((1, 17))

token_ids = torch.tensor(token_ids)
attention_mask = torch.tensor(attention_mask)

input_ids = [token_ids.cuda(), attention_mask.cuda()]


# outputs = model(token_ids, attention_mask)
outputs = model(input_ids)[-1]

out = outputs[-1]
print(out[0][0][11])
# print(len(outputs))
# data=open("attention.txt",'w+')
# print(outputs,file=data)
# data.close()
# print(attention)