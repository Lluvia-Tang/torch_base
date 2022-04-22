# -*- coding:utf-8 -*-
__author__ = 'tangs'
__time__ = '2021/11/29 18:54'

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split

# row_path = '/workspace/torch_base-main/data/covid-19-tweet/face_masks_all.csv'
row_path = './covid-19-tweet/face_masks_all.csv'
row_data = pd.read_csv(row_path)
'''
X = row_data["Text"].values
stance_y = row_data["Stance"].values
sentiment_y = row_data["Sentiment"].values

X_train, X_val, st_y_train, st_y_val = train_test_split(X, stance_y, test_size=0.4, random_state=45)

# train_data1 = np.column_stack((X_train,st_y_train))
train_data1 = np.column_stack((X_val,st_y_val))

X_train, X_val, sent_y_train, sent_y_val = train_test_split(X, sentiment_y, test_size=0.4, random_state=45)

# print(type(DataFrame(X_train)))
# train_data =  np.column_stack((train_data1,sent_y_train))
# train_data = DataFrame(train_data)
# train_data.to_csv("face_masks_train_64.csv",index = False)


train_data =  np.column_stack((train_data1,sent_y_val))
train_data = DataFrame(train_data)
train_data.to_csv("face_masks_test_64.csv",index = False)
print("已经生成了！")
'''
'''
stance_y = row_data["Stance"].values
sentiment_y = row_data["Sentiment"].values

st_y_train, st_y_val, sent_y_train, sent_y_val = train_test_split(stance_y, sentiment_y, test_size=0.4, random_state=45)
target_data = np.column_stack((st_y_train,sent_y_train))
# target_data = np.row_stack((st_y_train,sent_y_train))
# target_data = DataFrame(target_data)
# target_data.to_csv("covid-target.csv")
target_val = np.column_stack((st_y_val,sent_y_val))
target = np.row_stack((target_data,target_val))
target = DataFrame(target)

target.to_csv("covid-target2.csv")
'''
# save_name = "covid-target.txt"
# with open(save_name, mode='w') as fout:

path = "covid-target2.csv"
data = pd.read_csv(path)
list1 = ['train' for i in range(4080)]
list2 = ['test' for j in range(2721)]
list = list1 + list2
data.insert(1,'a',list)
data.to_csv("covid-target.csv",index = False)