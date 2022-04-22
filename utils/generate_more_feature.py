# -*- coding:utf-8 -*-
__author__ = 'tangs'
__time__ = '2021/12/16 15:56'

import os
import pickle

import numpy as np
import pandas as pd
import torch

from utils.clear import clean_str
'''
生成部分词性特征（根据hownet词典
和位置特征
'''
pos_s_dic_path = "/workspace/torch_base/data/dict/hownet/positive sentiment words.txt"
pos_e_dic_path = "/workspace/torch_base/data/dict/hownet/positive evaluation words.txt"
neg_s_dic_path = "/workspace/torch_base/data/dict/hownet/negative sentiment words.txt"
neg_e_dic_path = "/workspace/torch_base/data/dict/hownet/negative evaluation words.txt"
adv_path = "/workspace/torch_base/data/dict/hownet/adverb of degree.txt"

def write_to_list(path,dict):
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.strip("")
            dict.append(line.strip('\n'))
    return dict

pos_s_dic = list()
pos_s_dic = write_to_list(pos_s_dic_path,pos_s_dic)
pos_e_dic = list()
pos_e_dic = write_to_list(pos_e_dic_path,pos_e_dic)
neg_s_dic = list()
neg_s_dic = write_to_list(neg_s_dic_path,neg_s_dic)
neg_e_dic = list()
neg_e_dic = write_to_list(neg_e_dic_path,neg_e_dic)
adv_dic = list()
adv_dic = write_to_list(adv_path, adv_dic)

tag_dic = {"pos_s":1,"pos_e":2,"neg_s":3, "neg_e":4, "adv":5, "none":0}
position_dic = {"special":1,"common":0}

def process_idx(words_list): #index
    position = [0 for k in range(len(words_list))]
    tags = [0 for k in range(len(words_list))]
    for idx,word in enumerate(words_list):
        if word in pos_s_dic:
            tags[idx] = tag_dic["pos_s"]
            position[idx] = 1
        elif word in pos_e_dic:
            tags[idx] = tag_dic["pos_e"]
            position[idx] = 1
        elif word in neg_s_dic:
            tags[idx] = tag_dic["neg_s"]
            position[idx] = 1
        elif word in neg_e_dic:
            tags[idx] = tag_dic["neg_e"]
            position[idx] = 1
        elif word in adv_dic:
            tags[idx] = tag_dic["adv"]
            position[idx] = 1
        else:
            tags[idx] = 0
            position[idx] = 0

    return tags,position


def get_wordvec(word2id, vec_file_path, vec_dim=100):
    """ 读出txt文件的预训练词向量 """
    embedding_matrix_file_name = '{0}_{1}_embedding_matrix.pkl'.format(str(100), "position")
    # embedding_matrix_file_name = '{0}_{1}_embedding_matrix.pkl'.format(str(100), "position")
    if os.path.exists(embedding_matrix_file_name):
        print('开始加载词向量:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('loading word vectors ...')
        embedding_matrix = torch.nn.init.xavier_uniform_(torch.empty(len(word2id), vec_dim))
        embedding_matrix[0, :] = 0  # <pad>
        fname = '/root/files/glove.6B.100d.txt'
        with open(vec_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                splited = line.split(" ")
                if splited[0] in word2id:
                    embedding_matrix[word2id[splited[0]]] = torch.tensor(list(map(lambda x: float(x), splited[1:])))
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    # print("总共 %d个词，其中%d个找到了对应的词向量" % (len(word2id), found))
    return embedding_matrix.float()

word_vectors = get_wordvec(position_dic, vec_file_path='/root/files/glove.6B.100d.txt', vec_dim=100)

# if __name__ == '__main__':
    # process_data("/workspace/torch_base-main/data/covid-19-tweet/face_masks_train_64.csv")

    # 'train': '/workspace/torch_base-main/data/covid-19-tweet/face_masks_train_64.csv',
    # 'test': '/workspace/torch_base-main/data/covid-19-tweet/face_masks_test_64.csv'
