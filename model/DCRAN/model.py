# -*- coding:utf-8 -*-
__author__ = 'tangs'
__time__ = '2022/2/27 17:36'
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class DCRAN(nn.Module):
    def __init__(self, word_embedding, domain_embedding, word_dict):
        super(DCRAN, self).__init__()
        self.w2v = word_embedding
        self.w2v_domain = domain_embedding
