# -*- coding:utf-8 -*-
__author__ = 'tangs'
__time__ = '2021/12/16 19:58'

import os
import pickle

import torch


def get_wordvec(embedding_matrix_file_name):
    """ 读出txt文件的预训练词向量 """
    if os.path.exists(embedding_matrix_file_name):
        print('开始加载词向量:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('找不到 ...')

    return embedding_matrix.float()