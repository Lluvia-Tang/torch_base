# -*- coding:utf-8 -*-
__author__ = 'tangs'
__time__ = '2021/12/16 17:10'

import pickle

path = '../100_tag_embedding_matrix.pkl'  # path='/root/……/aus_openface.pkl'   pkl文件所在路径

f = open(path, 'rb')
data = pickle.load(f)

print(data)
print(len(data))