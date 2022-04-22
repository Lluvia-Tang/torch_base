# -*- coding:utf-8 -*-
__author__ = 'tangs'
__time__ = '2021/12/1 20:24'

import numpy as np
import torch
from torch import nn
from transformers import AdamW, BertForSequenceClassification, get_linear_schedule_with_warmup

import metrics
import sklearn
from transformers import BertConfig
from data.BERT import data_process
from model.BERT.base_BERT import BertForClassification
from paddlenlp.transformers import SkepForSequenceClassification, SkepTokenizer

device = "cuda:0"
model_name = "digitalepidemiologylab/covid-twitter-bert-v2"
# 指定模型名称，一键加载模型
model = SkepForSequenceClassification.from_pretrained(pretrained_model_name_or_path="skep_ernie_2.0_large_en", num_classes=3)
# 同样地，通过指定模型名称一键加载对应的Tokenizer，用于处理文本数据，如切分token，转token_id等。
tokenizer = SkepTokenizer.from_pretrained(pretrained_model_name_or_path="skep_ernie_2.0_large_en")
