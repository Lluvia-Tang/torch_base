# -*- coding:utf-8 -*-
__author__ = 'tangs'
__time__ = '2021/12/18 10:53'
# from senta import Senta

# my_senta = Senta()

# 选择是否使用gpu
use_cuda = True # 设置True or False

# 获取目前支持的情感预训练模型, 我们开放了以ERNIE 1.0 large(中文)、ERNIE 2.0 large(英文)和RoBERTa large(英文)作为初始化的SKEP模型
# print(my_senta.get_support_model()) # ["ernie_1.0_skep_large_ch", "ernie_2.0_skep_large_en", "roberta_skep_large_en"]

# 获取目前支持的预测任务
# print(my_senta.get_support_task()) # ["sentiment_classify", "aspect_sentiment_classify", "extraction"]

# my_senta.init_model(model_class="ernie_2.0_skep_large_en", task="sentiment_classify", use_cuda=use_cuda)
# texts = ["a sometimes tedious film ."]
# result = my_senta.predict(texts)
# print(result)

# my_senta.init_model(model_class="roberta_skep_large_en", task="sentiment_classify", use_cuda=use_cuda)
# texts = ["a sometimes tedious film ."]
# result = my_senta.predict(texts)
# print(result)
