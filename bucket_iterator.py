# -*- coding: utf-8 -*-
'''
填充数据迭代器
'''
import math
import random
import torch
import numpy
from data.make_datalodaer import DatesetReader


class BucketIterator(object):
    def __init__(self, data, batch_size, sort_key='text_indices', shuffle=True, sort=True):
        self.shuffle = shuffle
        self.sort = sort
        self.sort_key = sort_key
        self.batches = self.sort_and_pad(data, batch_size)
        self.batch_len = len(self.batches)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        if self.sort:
            # sorted_data = sorted(data, key=lambda x: len(x[self.sort_key]))
            sorted_data = sorted(data, key=lambda x: len(x[self.sort_key]), reverse=True)
        else:
            sorted_data = data
        batches = []
        for i in range(num_batch):
            batches.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size]))
        return batches

    def pad_data(self, batch_data):
        batch_text = []
        batch_target = []
        batch_text_indices = []
        batch_target_indices = []
        batch_stance = []
        batch_sentiment = []
        batch_attention_mask = []
        batch_tag_idx = []
        batch_position_idx = []
        batch_graph = []

        max_len = len(batch_data[0]['text_indices'])
        # print("max_len------------:",max_len)

        for item in batch_data:
            # text, target, text_indices, target_indices, stance, sentiment,tag_idx, position_idx, attention_mask = \
            #     item['text'], item['target'], item['text_indices'], item['target_indices'], \
            #     item['stance'], item['sentiment'], item['tag_idx'], item['position_idx'], item['attention_mask']

            # '''Bert时：'''
            text, target, text_indices, target_indices, stance, sentiment, attention_mask, graph = \
                item['text'], item['target'], item['text_indices'], item['target_indices'], \
                item['stance'], item['sentiment'], item['attention_mask'], item['graph']

            text_padding = [0] * (max_len - len(text_indices))
            target_padding = [0] * (max_len - len(target_indices))
            mask_padding = [0] * (max_len - len(attention_mask))
            # tag_padding = [0] * (max_len - len(tag_idx))
            # position_padding = [0] * (max_len - len(position_idx))

            batch_text.append(text)
            batch_target.append(target)

            batch_text_indices.append(text_indices + text_padding)
            # batch_target_indices.append(target_indices + target_padding)

            # batch_tag_idx.append(tag_idx + tag_padding)
            # batch_position_idx.append(position_idx + position_padding)

            batch_target_indices.append(target_indices)
            batch_stance.append(stance)
            batch_sentiment.append(sentiment)
            batch_attention_mask.append(attention_mask + mask_padding)
            batch_graph.append(numpy.pad(graph, ((0, max_len-len(text_indices)), (0, max_len-len(text_indices))), 'constant'))

        return {
                'text': batch_text,
                'target': batch_target,
                'text_indices': torch.tensor(batch_text_indices),
                'target_indices': torch.tensor(batch_target_indices),
                'stance': torch.tensor(batch_stance),
                'sentiment': torch.tensor(batch_sentiment),
                'graph': torch.tensor(batch_graph),
                'max_len' : torch.tensor(max_len),
                'tag_indices' : torch.tensor(batch_tag_idx),
                'position_indices' : torch.tensor(batch_position_idx),
                'attention_mask':torch.tensor(batch_attention_mask)
            }

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for idx in range(self.batch_len):
            yield self.batches[idx]

if __name__ == '__main__':
    data = DatesetReader(dataset='covid-19-tweet', embed_dim=300)
    train_data_loader = BucketIterator(data=data.train_data, batch_size=16, shuffle=True)
    test_data_loader = BucketIterator(data=data.test_data, batch_size=16, shuffle=True)

    for i_batch, sample_batched in enumerate(train_data_loader):
        print(len(sample_batched))
        print(sample_batched.keys())
        for k, v in sample_batched.items():
            print(k, v)
            if k not in ['text', 'target']:
                print(v.size())

        break


