# -*- coding:utf-8 -*-

__time__ = '2021/11/30 9:32'

import re

def preprocessing_tweet(text):
    #对推文进行预处理
    # reference: https://github.com/zhouyiwei/tsd/blob/e1db26a829f8702f437accd42a998ce8e9344de1/utils.py#L5
    text = re.sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<URL>", text)
    text = re.sub(r"@\w+", "<USER>", text)
    text = re.sub(r"[8:=;]['`\-]?[)d]+|[)d]+['`\-]?[8:=;]", "<SMILE>", text)
    text = re.sub(r"[8:=;]['`\-]?p+", "<LOLFACE>", text)
    text = re.sub(r"[8:=;]['`\-]?\(+|\)+['`\-]?[8:=;]", "<SADFACE>", text)
    text = re.sub(r"[8:=;]['`\-]?[\/|l*]", "<NEUTRALFACE>", text)
    text = re.sub(r"<3","<HEART>", text)
    text = re.sub(r"/"," / ", text)
    text = re.sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<NUMBER>", text)
    p = re.compile(r"#\S+")
    text = p.sub(lambda s: "<HASHTAG> "+s.group()+" <ALLCAPS>"
                 if s.group()[1:].isupper()
                 else " ".join(["<HASHTAG>"]+re.split(r"([A-Z][^A-Z]*)", s.group()[1:])), text)
    text = re.sub(r"([!?.]){2,}", r"\1 <REPEAT>", text)
    text = re.sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <ELONG>", text)

    return text.lower()