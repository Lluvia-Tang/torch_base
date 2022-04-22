# -*- coding:utf-8 -*-
__author__ = 'tangs'
__time__ = '2021/12/6 0:03'

import spacy
from utils.clear import clean_str
import pandas as pd

nlp = spacy.load('en_core_web_sm')
row_path = '../data/covid-19-tweet/face_masks_all.csv'

data = pd.read_csv(row_path)
text_ilst = data["Text"].values

save_name = 'covid.txt'

with open(save_name, mode='w') as fout:
    for text in text_ilst:
        text = clean_str(text)
        doc = nlp(text)
        word_list = [str(token) for token in doc if not token.is_punct | token.is_stop]
        txt = ' '.join(word_list)

        fout.write(txt)
        fout.write(" \n")

