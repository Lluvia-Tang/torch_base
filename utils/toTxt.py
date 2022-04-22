# -*- coding:utf-8 -*-
__author__ = 'tangs'
__time__ = '2021/12/6 19:08'

import pandas as pd
import os

data = pd.read_csv('../data/covid-19-tweet/covid-target.csv',header=None, encoding='utf-8')
with open('covid.txt', 'a+', encoding='utf-8') as f:
    for line in data.values:
        f.write((str(line[0]) + '\t' + str(line[1]) +'\t' + str(line[2])+'\t' + str(line[3])+ '\n'))
