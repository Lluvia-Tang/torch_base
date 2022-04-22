# -*- coding:utf-8 -*-
__author__ = 'tangs'
__time__ = '2021/12/6 10:37'
import networkx as nx
import numpy as np
import pandas as pd
import torch as th
from sklearn.model_selection import train_test_split

from utils.torch_utils import preprocess_adj

def read_file(path, mode='r', encoding=None):
    if mode not in {"r", "rb"}:
        raise ValueError("only read")
    return open(path, mode=mode, encoding=encoding)

def get_train_test(target_fn):
    train_lst = list()
    test_lst = list()
    with read_file(target_fn, mode="r") as fin:
        for indx, item in enumerate(fin):
            if item.split("\t")[1] in {"train", "training", "20news-bydate-train"}:
                train_lst.append(indx)
            else:
                test_lst.append(indx)

    return train_lst, test_lst

class PrepareData:
    def __init__(self):
        print("prepare data")
        self.graph_path = "/workspace/torch_base-main/data/graph"
        self.dataset = "covid"

        # graph
        graph = nx.read_weighted_edgelist(f"{self.graph_path}/{self.dataset}.txt"
                                          , nodetype=int)
        # print_graph_detail(graph)
        adj = nx.to_scipy_sparse_matrix(graph,
                                        nodelist=list(range(graph.number_of_nodes())),
                                        weight='weight',
                                        dtype=np.float)

        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        self.adj = preprocess_adj(adj, is_sparse=True)

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # features
        self.nfeat_dim = graph.number_of_nodes()  #特征数量
        row = list(range(self.nfeat_dim))
        col = list(range(self.nfeat_dim))
        value = [1.] * self.nfeat_dim
        shape = (self.nfeat_dim, self.nfeat_dim)
        indices = th.from_numpy(
                np.vstack((row, col)).astype(np.int64))
        values = th.FloatTensor(value)
        shape = th.Size(shape)

        self.features = th.sparse.FloatTensor(indices, values, shape)

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # target

        stance_label_dic = {'FAVOR': 1, 'AGAINST': 0, 'NONE': 2}
        senti_label_dic = {'Positive': 1, 'Negative': 0, 'Other': 2}
        # row_path = '/workspace/torch_base-main/data/covid-19-tweet/face_masks_all.csv',
        target_fn = "/workspace/torch_base-main/data/covid-19-tweet/covid-target.csv"
        data = pd.read_csv(target_fn,header=None)
        target = list()
        for item in data.values:
            target.append(item[2])      # stance label

        # target2id = {label: indx for indx, label in enumerate(set(target))}
        self.target = [stance_label_dic[label] for label in target]  # 得到index类型的label列表

        self.nclass = 3 #类别数

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # train val test split

        # self.train_lst, self.test_lst = get_train_test(target_fn)  #训练集测试集index列表


        train_lst = list()
        test_lst = list()
        for item in data.values:
            if item[1] == 'train':
                train_lst.append(item[0])
            else:
                test_lst.append(item[0])
        self.train_lst = train_lst
        self.test_lst = test_lst


#predata = PrepareData()