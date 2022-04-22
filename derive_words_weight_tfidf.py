# -*- coding: utf-8 -*-

import numpy as np
import spacy
import pickle as pkl
import nltk
import math
from collections import Counter
import pandas as pd
from utils.clear import clean_str

nlp = spacy.load('en_core_web_sm')

# TARGET_DIC = {'Climate Change is a Real Concern': 0, 'Atheism': 1,
#               'Feminist Movement': 2, 'Hillary Clinton': 3, 'Legalization of Abortion': 4}
# TARGET_DIC = {'Legalization of Abortion': 0,'legalization of abortion':1}
INDEX = 0
stance_label_dic = {'FAVOR': 1,'AGAINST': 0, 'NONE': 2}
senti_label_dic = {'pos' : 1, 'Positive' : 1, 'POSITIVE' : 1, 'neg': 0, 'Negative': 0, 'NEGATIVE' : 0, 'Other': 2, 'other': 2 ,'NEITHER' : 2}


def generate_word_weight_tfidf(filename):

    def tf(word, count):
        return count[word] / sum(count.values())

    def n_containing(word, count_list):
        return sum(1 for count in count_list if word in count)

    def idf(word, count_list):
        return math.log(len(count_list) / (1 + n_containing(word, count_list)))

    def tfidf(word, count, count_list):
        return tf(word, count) * idf(word, count_list)

    word_list = []
    countlist = []
    word_in_document_tfidf = []

    df = pd.read_csv(filename)
    for line in df.values:
        text = line[0]
        target = line[1]
        sent = clean_str(text)  # 对推文文本进行处理
        # vocab.update(sent.split(" "))
        doc = nlp(sent)  # 使用spacy分词
        indices = [str(token) for token in doc]
        text = ' '.join(indices).lower()
        word_list.append(text.split(' '))

    for i in range(len(word_list)):
        count = Counter(word_list[i])
        countlist.append(count)
    for i, count in enumerate(countlist):
        # print("Top words in document {}".format(i + 1))
        scores = {word: tfidf(word, count, countlist) for word in count}
        word_in_document_tfidf.append(scores)
        # print(word_in_document_tfidf)
        # sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        # for word, score in sorted_words[:]:
        #     print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))
    path = filename.replace('.csv', '.word.weight.tfidf')
    path = path.replace('data/covid-19-tweet', 'pre_data')
    # path = filename.replace()"./pre_data/face_masks_train.word.weight.tfidf"
    with open(path, 'wb') as fout:
        pkl.dump(word_in_document_tfidf, fout)
    print('done !!!' + path)

    # return word_in_document_tfidf

def load_seed_word(path):
    seed_words = {}
    fp = open(path, 'r')
    for line in fp:
        line = line.strip()
        if not line:
            continue
        word, weight = line.split('\t')
        seed_words[word] = weight
    fp.close()
    return seed_words


def dependency_adj_matrix(text, seed_words):
    # https://spacy.io/docs/usage/processing-text
    document = nlp(text)
    seq_len = len(text.split())
    matrix = np.zeros((seq_len, seq_len)).astype('float32')

    for token in document:
        if str(token) in seed_words:
            weight = float(seed_words[str(token)])
        else:
            weight = 0
        if token.i < seq_len:
            matrix[token.i][token.i] = 1 + weight
            # https://spacy.io/docs/api/token
            for child in token.children:
                if str(child) in seed_words:
                    weight += float(seed_words[str(child)])
                if child.i < seq_len:
                    matrix[token.i][child.i] = 1 + weight
                    matrix[child.i][token.i] = 1 + weight

    # print(matrix)
    # print('='*30)
    return matrix

#生成每个target的raw
def generate_raw(filename): #csv
    df = pd.read_csv(filename)
    path = filename.replace(".csv",".raw")
    path = path.replace('data/covid-19-tweet', 'pre_data')
    # path = "./pre_data/face_masks_train.raw"
    fout = open(path, 'w', encoding='utf-8')
    for line in df.values:
        text = line[0]
        target = line[1]
        stance = stance_label_dic[line[2]]
        senti = senti_label_dic[line[3]]
        sent = clean_str(text)  # 对推文文本进行处理
        doc = nlp(sent)  # 使用spacy分词
        indices = [str(token) for token in doc]
        text = ' '.join(indices).lower()
        fout.write(text+"\n")
        fout.write(target+"\n")
        fout.write(str(stance)+"\n")
        fout.write(str(senti)+"\n") #多一行需要删除
    fout.close()
    print("done!!!"+path)


def process(filename, word_weight_file): #filename: .raw
    idx2graph = {}
    graph_idx = 0
    with open(word_weight_file, 'rb') as f:  # "face_masks_train.word.weight.tfidf"
        document_word_weight = pkl.load(f)

    with open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
        lines = fin.readlines()
        #print("len(all.sttrain.orig.raw)",len(lines))
        for index,i in enumerate(range(0, len(lines), 4)):
            text = lines[i].lower().strip()
            target = lines[i + 1].lower().strip()
            adj_matrix = dependency_adj_matrix(text, document_word_weight[index])
            idx2graph[graph_idx] = adj_matrix
            graph_idx = graph_idx + 1

    print(idx2graph)
    path = filename.replace(".raw", ".graph.tfidf.graph")
    #with open(filename.replace("orig.raw", "graph.tfidf") + '.graph', 'wb') as fout:
    with open(path, 'wb') as fout:
        pkl.dump(idx2graph, fout)
    print('done !!!' + path)


if __name__ == '__main__':
    ##生成tfidf图
    # generate_word_weight_tfidf("/workspace/torch_base/data/covid-19-tweet/face_masks_train_64.csv")
    # generate_word_weight_tfidf("/workspace/torch_base/data/covid-19-tweet/face_masks_test_64.csv")
    # generate_raw("/workspace/torch_base/data/covid-19-tweet/face_masks_train_64.csv")
    # generate_raw("/workspace/torch_base/data/covid-19-tweet/face_masks_test_64.csv")
    # process("/workspace/torch_base/pre_data/face_masks_train_64.raw", '/workspace/torch_base/pre_data/face_masks_train_64.word.weight.tfidf')
    # process("/workspace/torch_base/pre_data/face_masks_test_64.raw", '/workspace/torch_base/pre_data/face_masks_test_64.word.weight.tfidf')


    #with open('./raw_data/all.train.graph.tfidf.graph', 'rb') as f:
    with open('./pre_data/face_masks_test_64.graph.tfidf.graph', 'rb') as f:
        idx2graph = pkl.load(f)
    print(len(idx2graph))
    print(idx2graph[0])
