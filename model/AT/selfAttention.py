# -*- coding:utf-8 -*-
__author__ = 'tangs'
__time__ = '2021/12/17 11:36'
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

'''
自注意力
attention_dim = args.hidden if not args.bi else 2*args.hidden
attention = Attention(attention_dim, attention_dim, attention_dim)
'''

class Attention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim):
        super(Attention, self).__init__()
        self.scale = 1. / math.sqrt(query_dim)

    def forward(self, query, keys, values):
        # Query = [BxQ]
        # Keys = [TxBxK]
        # Values = [TxBxV]
        # Outputs = a:[TxB], lin_comb:[BxV]

        # Here we assume q_dim == k_dim (dot product attention)

        query = query.unsqueeze(1) # [BxQ] -> [Bx1xQ]
        keys = keys.transpose(0,1).transpose(1,2) # [TxBxK] -> [BxKxT]
        energy = torch.bmm(query, keys) # [Bx1xQ]x[BxKxT] -> [Bx1xT]
        energy = F.softmax(energy.mul_(self.scale), dim=2) # scale, normalize

        values = values.transpose(0,1) # [TxBxV] -> [BxTxV]
        linear_combination = torch.bmm(energy, values).squeeze(1) #[Bx1xT]x[BxTxV] -> [BxV]
        return energy, linear_combination


'''
    outputs, (hidden) = self.encoder(self.embedding(input))
    energy, linear_combination = self.attention(hidden, outputs, outputs) 
    logits = self.decoder(linear_combination)【nn.Linear(hidden_dim, num_classes)】
'''

