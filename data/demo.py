# -*- coding:utf-8 -*-
__author__ = 'tangs'
__time__ = '2021/11/29 15:44'

import pickle as pkl
import random

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
import torch
import gzip
import csv
import matplotlib.pyplot as plt
import numpy as np
import time
import math
import pandas as pd
np.set_printoptions(threshold=np.inf)
# from utils.clear import clean_str
# sentence = []
# with open("/workspace/torch_base/data/clean_corpus/covid.txt", 'r') as f:
#     while True:
#         fa = f.readline().strip()
#         print(fa)
#         if fa == "":
#             break
#         sentence.append(fa)
#
# print(sentence)
# print(len(sentence))

# df = pd.read_csv("/workspace/torch_base/data/covid-19-tweet/fauci_train.csv")
# # for line in df.values:
# #     print(line)
# #     break
# sentence = df.loc[3, 'Text']#将3改为index可以定位到每一行的一句话
# print(sentence)
# a = "a"
# list = ["a","b","c","d"]
# # print(a in list)
# position = [0 for k in range(len(list))]
# position[1] = 1
# print(position)
# 'train': '/workspace/torch_base-main/data/covid-19-tweet/face_masks_train_64.csv',
# 'test': '/workspace/torch_base-main/data/covid-19-tweet/face_masks_test_64.csv'
#
# senti_label_dic = {'pos' : 1, 'Positive' : 1,'neg': 0, 'Negative': 0, 'Other': 2, 'other': 2}
# fpath = "dev.txt"
# with open(fpath, 'w') as f:
#     df = pd.read_csv("/workspace/torch_base/data/covid-19-tweet/face_masks_test_64.csv")
#     for line in df.values:
#         text = line[0]
#         text = clean_str(text)
#         f.write(str(senti_label_dic[line[2]])+"	"+text+"\n")

# h1 = torch.Tensor([[[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]],[[1,1,1,1,1],[2,2,2,2,2],[3,3,3,3,3]]])
#
# h2 = torch.Tensor([[[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]],[[1,1,1,1,1],[2,2,2,2,2],[3,3,3,3,3]]])
# print(h1.shape) #torch.Size([2, 3, 5]) [n_layers * 2, batch_size, self.hidden_size]
# input = torch.Tensor([[1,0,0,0],[2,0,0,0],[3,0,0,0]]) #B x Seqlen
# b = torch.Tensor(np.zeros([3,1]))
# input = torch.cat((input,b),dim=1)
# print(input.shape)
# print(input)
# input = input.unsqueeze(dim=0)
# input = input.expand(2,3,4)
#
# # print(input.shape)
# # print(input.shape)   #torch.Size([2, 3, 4])
# a = torch.cat([h1,input],dim=2)
# print(a.shape)  #torch.Size([2, 3, 9])
# h = h1.mul(h2)
# print(h)
# print(h.shape)


# path = 'a.word.weight.tfidf'
# # path = filename.replace()"./pre_data/face_masks_train.word.weight.tfidf"
# with open(path, 'wb') as fout:
#     pkl.dump('a', fout)
# print('done !!!' + path)

# path = "/workspace/torch_base/pre_data/face_masks_train_64.graph.tfidf.graph"
# with open(path, 'rb') as f:
#     idx2graph = pkl.load(f)
# print(len(idx2graph))

# a = [1,2,4,5,5]
# print(a[:-1])

# from transformers import BertTokenizer
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# bert_input_per = tokenizer.convert_tokens_to_ids("[MASK]")
# print(bert_input_per)  #[CLS]101  [SEP]102  [MASK]103

# print(torch.argmax(torch.Tensor([[1, 0, 0],[0, 1, 0], [0, 0, 1]]), dim=1))

# a= [1,2,3]
# b = a.copy()
# b[2] = 5
# print(a, b)

# def get_annotated_aope_targets(sents, labels):
# #     annotated_targets = []
# #     num_sents = len(sents)
# #     for i in range(num_sents):
# #         tuples = labels[i]
# #         # tup: ([3, 4], [2])
# #         for tup in tuples:
# #             ap, op = tup[0], tup[1]
# #             opt = [sents[i][j] for j in op]
# #             # multiple OT for one AP
# #             if '[' in sents[i][ap[0]]:
# #                 if len(ap) == 1:
# #                     sents[i][ap[0]] = f"{sents[i][ap[0]][:-1]}, {' '.join(opt)}]"
# #                 else:
# #                     sents[i][ap[-1]] = f"{sents[i][ap[-1]][:-1]}, {' '.join(opt)}]"
# #             else:
# #                 annotation = f"{' '.join(opt)}"
# #                 if len(ap) == 1:
# #                     sents[i][ap[0]] = f"[{sents[i][ap[0]]}|{annotation}]"
# #                 else:
# #                     sents[i][ap[0]] = f"[{sents[i][ap[0]]}"
# #                     sents[i][ap[-1]] = f"{sents[i][ap[-1]]}|{annotation}]"
# #         annotated_targets.append(sents[i])
# #
# #     return annotated_targets
# #
# # sents = [["It","'s", "fast", "and", "has", "excellent", "battery", "life", "."]]
# # labels = [[([6, 7], [5])]]
# # targets = get_annotated_aope_targets(sents, labels)
# # print(targets)

# direct = False
# a = 0
# if not direct:
#     a = 1+1
# print(a)
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
print('Tokenized: ', tokenizer.tokenize("the chicken is awful but for service it really makes me impeccable ."))