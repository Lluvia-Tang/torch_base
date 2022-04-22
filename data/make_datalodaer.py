# -*- coding:utf-8 -*-
__author__ = 'tangs'
__time__ = '2021/11/29 14:28'
'''
using spacy+glove 
'''

import pickle
from utils.generate_more_feature import process_idx
import pandas as pd
import os
import pickle as pkl
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset,DataLoader
from utils.pre_data import preprocessing_tweet
import spacy
from utils.clear import clean_str


stance_label_dic = {'FAVOR': 1,'AGAINST': 0, 'NONE': 2}
senti_label_dic = {'pos' : 1, 'Positive' : 1, 'POSITIVE' : 1, 'neg': 0, 'Negative': 0, 'NEGATIVE' : 0, 'Other': 2, 'other': 2 ,'NEITHER' : 2}
filename = '/root/files/glove.42B.300d.txt'

# row_path = '/workspace/torch_base/data/covid-19-tweet/fauci_all.csv'
# row_path = '/workspace/torch_base/data/covid-19-tweet/school_closures_all.csv'
# row_path = '/workspace/torch_base/data/covid-19-tweet/stay_at_home_orders_all.csv'
# row_path = '/workspace/torch_base/data/covid-19-tweet/covid_19_all.csv'
# row_path = '/workspace/torch_base/data/covid-19-tweet/pre_face_masks_all.csv'
# row_path = '/workspace/torch_base/data/semEval2016/semEval_abortion_all.csv'
row_path = '/workspace/torch_base/data/semEval2016/semEval_all.csv'


# USE_GPU = True
nlp = spacy.load('en_core_web_sm')

target = "face masks"

class Language:
    """ 根据句子列表建立词典并将单词列表转换为数值型表示 """
    def __init__(self):
        self.word2id = {}
        self.id2word = {}
        self.target = target

    def fit(self, sent_list):
        vocab = set()
        for sent in sent_list:
            # sent = preprocessing_tweet(sent)  #对推文文本进行处理
            sent = clean_str(sent)  #对推文文本进行处理
            # vocab.update(sent.split(" "))
            doc = nlp(sent)   #使用spacy分词
            vocab.update([str(token) for token in doc])     #使用spacy分词
            # vocab.update([str(token) for token in doc if not token.is_punct | token.is_stop])     #使用spacy分词
            # print(nlp(sent))
        #处理target
        # target = preprocessing_tweet(self.target)
        target = clean_str(self.target)
        vocab.update(target.split(" "))

        word_list = ["<pad>", "<unk>"] + list(vocab)
        # print("word_list:\n",word_list)
        self.word2id = {word: i for i, word in enumerate(word_list)}
        self.id2word = {i: word for i, word in enumerate(word_list)}

    def transform(self, sent_list, reverse=False):
        sent_list_id = []
        word_mapper = self.word2id if not reverse else self.id2word
        unk = self.word2id["<unk>"] if not reverse else None
        for sent in sent_list:
            # sent = preprocessing_tweet(sent)
            sent = clean_str(sent)
            sent_id = list(map(lambda x: word_mapper.get(x, unk), sent.split(" ") if not reverse else sent))
            sent_list_id.append(sent_id)
        # print("sent_list_id:\n",sent_list_id)

        # target = preprocessing_tweet(self.target)
        target = clean_str(self.target)
        target_id = list(map(lambda x: word_mapper.get(x, unk), target.split(" ") if not reverse else target))

        return sent_list_id, target_id

def get_wordvec(word2id, vec_file_path, vec_dim=300):
    """ 读出txt文件的预训练词向量 """
    # embedding_matrix_file_name = '{0}_{1}_embedding_matrix.pkl'.format(str(300), "covid-19_fauci")
    # embedding_matrix_file_name = '{0}_{1}_embedding_matrix.pkl'.format(str(300), "covid-19_school_closures")
    # embedding_matrix_file_name = '{0}_{1}_embedding_matrix.pkl'.format(str(300), "covid-19_all")
    embedding_matrix_file_name = '{0}_{1}_embedding_matrix.pkl'.format(str(300), "semEval_all")
    # embedding_matrix_file_name = '{0}_{1}_embedding_matrix.pkl'.format(str(300), "semEval_abortion")

    if os.path.exists(embedding_matrix_file_name):
        print('开始加载词向量:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('loading word vectors ...')
        embedding_matrix = torch.nn.init.xavier_uniform_(torch.empty(len(word2id), vec_dim))
        embedding_matrix[0, :] = 0  # <pad>
        fname = '/root/files/glove.42B.300d.txt'
        with open(vec_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                splited = line.split(" ")
                if splited[0] in word2id:
                    embedding_matrix[word2id[splited[0]]] = torch.tensor(list(map(lambda x: float(x), splited[1:])))
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    # print("总共 %d个词，其中%d个找到了对应的词向量" % (len(word2id), found))
    return embedding_matrix.float()


class TextDataset(object):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class DatesetReader:

    @staticmethod
    def __read_data__(fname,X_language):

        df = pd.read_csv(fname)

        all_data = []
        print("read_data: ",fname)

        # cnt = 0

        # sent_list = []
        # with open(cleaned_text, 'r') as f:
        #     while True:
        #         fa = f.readline()
        #         if fa == "":
        #             break
        #         sent_list.append(fa)

        #for i in range(0, len(lines), 4):
        # # for row in df.itertuples():
        #     text = getattr(row, 'Text').lower().strip()
        #     stance = getattr(row, 'Stance')
        #     senti = getattr(row, 'Sentiment')

        fname = fname.replace(".csv", ".graph.tfidf.graph")
        path = fname.replace('data/covid-19-tweet', 'pre_data')
        print("loading graph from :",path)
        with open(path, 'rb') as fin:
            idx2graph = pkl.load(fin)
        graph_idx = 0

        for line in df.values:
            text = line[0]
            stance = stance_label_dic[line[2]]
            senti = senti_label_dic[line[3]]
            # stance = stance_label_dic[line[1]]
            # senti = senti_label_dic[line[2]]
            # print("senti: \n",senti)
            graph = idx2graph[graph_idx]
            graph_idx = graph_idx + 1

            word_mapper = X_language.word2id
            unk = X_language.word2id["<unk>"]

            # text = preprocessing_tweet(text)
            text = clean_str(text)
            doc = nlp(text)
            # text_indices = list(map(lambda x: word_mapper.get(x, unk), text.split(" ")))
            # [str(token) for token in doc if not token.is_punct | token.is_stop]
            # text_indices = []
            # indices = [str(token) for token in doc if not token.is_punct | token.is_stop]
            indices = [str(token) for token in doc]
            text_indices = list(map(lambda x: word_mapper.get(x, unk), indices))

            #加入tags和positions（multi-channel）
            tag_indices,position_indices = process_idx(indices)

            # for token in doc:
            #     if not token.is_punct | token.is_stop:
            # text_indices = list(map(lambda x: word_mapper.get(x, unk), str(token) for token in doc if not token.is_punct | token.is_stop))
            # print("sent_list_id:\n",sent_list_id)
            # target = preprocessing_tweet("face_masks")
            target = clean_str("face masks")
            target_id = list(map(lambda x: word_mapper.get(x, unk), target.split(" ")))

            #封装 attention_mask
            attention_mask = [float(i > 0) for i in text_indices]

            # text_indices = tokenizer.text_to_sequence(text)
            # target_indices = tokenizer.text_to_sequence(target)
            # graph = idx2graph[i]  #idx2graph:length 2914
            data = {
                'text': text,
                'target': target,
                'text_indices': text_indices,
                'target_indices': target_id,
                'stance': stance,
                'sentiment': senti,
                'tag_idx': tag_indices,
                'position_idx': position_indices,
                'attention_mask': attention_mask,
                'graph': graph,
            }

            all_data.append(data)
            # cnt += 1
        return all_data

    def __init__(self, dataset='covid-19-tweet', embed_dim=300):
        print("preparing {0} dataset ...".format(dataset)) #此处可以放入其它数据集等

        fname = {
            'covid-19-tweet': {
                'train': '/workspace/torch_base/data/covid-19-tweet/face_masks_train_64.csv',
                'test': '/workspace/torch_base/data/covid-19-tweet/face_masks_test_64.csv'
                # 'train': '/workspace/torch_base/data/covid-19-tweet/fauci_train.csv',
                # 'test': '/workspace/torch_base/data/covid-19-tweet/fauci_test.csv'
                # 'train': '/workspace/torch_base/data/covid-19-tweet/school_closures_train.csv',
                # 'test': '/workspace/torch_base/data/covid-19-tweet/school_closures_test.csv'
                # 'train': '/workspace/torch_base/data/covid-19-tweet/stay_at_home_orders_train.csv',
                # 'test': '/workspace/torch_base/data/covid-19-tweet/stay_at_home_orders_test.csv',
                # 'train': '/workspace/torch_base/data/covid-19-tweet/covid_19_train.csv',
                # 'test': '/workspace/torch_base/data/covid-19-tweet/covid_19_test.csv'
                # 'train': '/workspace/torch_base/data/covid-19-tweet/train.csv'
                # 'test': '/workspace/torch_base/data/covid-19-tweet/test.csv'
            },
            'semEval2016':{
                'train': '/workspace/torch_base/data/semEval2016/semEval_train.csv',
                # 'test': '/workspace/torch_base/data/semEval2016/semEval_test.csv'
                # 'train': '/workspace/torch_base/data/semEval2016/semEval_abortion_train.csv',
                'test': '/workspace/torch_base/data/semEval2016/semEval_climate_test.csv'
            }
        }

        df = pd.read_csv(row_path)
        # self.sentence = []
        # with open("/workspace/torch_base/data/clean_corpus/covid.txt",'r') as f:
        #     while True:
        #         fa = f.readline()
        #         if fa == "":
        #             break
        #         self.sentence.append(fa)

        # print(len(self.sentence))
        self.sentence = df["Text"].values
        self.X_language = Language()

        self.X_language.fit(self.sentence)
        # X, Target = X_language.transform(self.sentence)
        word_vectors = get_wordvec(self.X_language.word2id, vec_file_path=filename, vec_dim=300)

        self.embedding_matrix = word_vectors
        print("开始准备数据集....")

        # cleaned_train = "/workspace/torch_base/data/clean_corpus/train_clean.txt"
        # cleaned_test = "/workspace/torch_base/data/clean_corpus/test_clean.txt"

        self.train_data = TextDataset(DatesetReader.__read_data__(fname[dataset]['train'],self.X_language))
        self.test_data = TextDataset(DatesetReader.__read_data__(fname[dataset]['test'],self.X_language))
