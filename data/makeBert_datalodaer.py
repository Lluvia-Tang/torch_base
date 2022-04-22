# -*- coding:utf-8 -*-
__author__ = 'tangs'
__time__ = '2021/11/29 14:28'

from transformers import BertTokenizer

from options import prepare_train_args

'''
using spacy+bert
'''

import pickle

import pandas as pd
import os
import pickle as pkl
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset,DataLoader
from utils.pre_data import preprocessing_tweet
from utils.clear import clean_str
import spacy
# from paddlenlp.transformers import SkepForSequenceClassification, SkepTokenizer

stance_label_dic = {'FAVOR': 1,'AGAINST': 0, 'NONE': 2}
senti_label_dic = {'pos' : 1, 'Positive' : 1, 'POSITIVE' : 1, 'neg': 0, 'Negative': 0, 'NEGATIVE' : 0, 'Other': 2, 'other': 2 ,'NEITHER' : 2}
filename = '/root/files/glove.42B.300d.txt'
# row_path = '/workspace/torch_base/data/semEval2016/semEval_abortion_all.csv'
# row_path = '/workspace/torch_base-main/data/covid-19-tweet/pre_face_masks_all.csv'
# row_path = '/workspace/torch_base-main/data/covid-19-tweet/face_masks_semi.csv'


# USE_GPU = True
nlp = spacy.load('en_core_web_sm')

# train_data = pd.read_csv(train_path)
# test_data = pd.read_csv(test_path)

target = "face masks"
SEP,CLS='[SEP]','[CLS]'

class TextDataset(object):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class DatesetReader:

    @staticmethod
    def __read_data__(fname,tokenizer):

        df = pd.read_csv(fname)
        target = "face masks"
        all_data = []
        print("read_data: ",fname)
        # pad_size = 100

        # cnt = 0
        # sent_list = []
        # with open(cleaned_text, 'r') as f:
        #     while True:
        #         fa = f.readline()
        #         if fa == "":
        #             break
        #         sent_list.append(fa)
        fname = fname.replace(".csv", ".graph.tfidf.graph")
        path = fname.replace('data/covid-19-tweet', 'pre_data')

        with open(path, 'rb') as fin:
            idx2graph = pkl.load(fin)
        graph_idx = 0

        for line in df.values:
            text = line[0]
            stance = stance_label_dic[line[2]]
            senti = senti_label_dic[line[3]]
            # print("senti: \n",senti)
            graph = idx2graph[graph_idx]
            graph_idx = graph_idx + 1

            # text = preprocessing_tweet(text)
            text = clean_str(text)
            doc = nlp(text)
            doc = [str(token) for token in doc]
            # doc = [str(token) for token in doc if not token.is_punct | token.is_stop]
            token = [CLS] + doc + [SEP] #在序列前加一个[CLS]标志位
            # seq_len = len(token)

            token_ids = tokenizer.convert_tokens_to_ids(token)


            # target = preprocessing_tweet(target)
            target = clean_str(target)
            t = nlp(target)
            # t = [str(token) for token in t if not token.is_punct | token.is_stop]
            t = [str(token) for token in t]
            target_id = tokenizer.convert_tokens_to_ids(t)

            attention_mask = [float(i > 0) for i in token_ids]

            data = {
                'text': text,
                'target': target,
                'text_indices': token_ids,
                'target_indices': target_id,
                'stance': stance,
                'sentiment': senti,
                'attention_mask': attention_mask,
                'graph': graph,
            }

            all_data.append(data)
            # cnt += 1
        return all_data

    def __init__(self, dataset='covid-19-tweet'):
        print("preparing {0} dataset ...".format(dataset)) #此处可以放入其它数据集等

        fname = {
            'covid-19-tweet': {
                'train': '/workspace/torch_base/data/covid-19-tweet/face_masks_train_64.csv',
                'test': '/workspace/torch_base/data/covid-19-tweet/face_masks_test_64.csv'
                # 'train': '/workspace/torch_base/data/covid-19-tweet/stay_at_home_orders_train.csv',
                # 'test': '/workspace/torch_base/data/covid-19-tweet/stay_at_home_orders_test.csv'
                # 'train': '/workspace/torch_base/data/covid-19-tweet/school_closures_train.csv',
                # 'test': '/workspace/torch_base/data/covid-19-tweet/school_closures_test.csv'
                # 'train': '/workspace/torch_base/data/covid-19-tweet/fauci_train.csv',
                # 'test': '/workspace/torch_base/data/covid-19-tweet/fauci_test.csv'
                # 'train': '/workspace/torch_base/data/covid-19-tweet/covid_19_train.csv',
                # 'test': '/workspace/torch_base/data/covid-19-tweet/covid_19_test.csv'
                # 'train': '/workspace/torch_base/data/covid-19-tweet/train.csv',
                # 'test': '/workspace/torch_base/data/covid-19-tweet/test.csv'
            },
            'semEval2016': {
                # 'train': '/workspace/torch_base/data/semEval2016/semEval_climate_train.csv',
                # 'test': '/workspace/torch_base/data/semEval2016/semEval_climate_test.csv'
                'train': '/workspace/torch_base/data/semEval2016/semEval_train.csv',
                'test': '/workspace/torch_base/data/semEval2016/semEval_test.csv'
            }
        }

        args = prepare_train_args()
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(self.args.bert_path)
        # self.tokenizer = SkepTokenizer.from_pretrained(self.args.bert_path)

        # cleaned_train = "/workspace/torch_base/data/clean_corpus/train_clean.txt"
        # cleaned_test = "/workspace/torch_base/data/clean_corpus/test_clean.txt"

        print("开始装填数据....")
        self.train_data = TextDataset(DatesetReader.__read_data__(fname[dataset]['train'],self.tokenizer))
        self.test_data = TextDataset(DatesetReader.__read_data__(fname[dataset]['test'],self.tokenizer))


