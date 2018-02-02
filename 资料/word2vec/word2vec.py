#! /usr/bin/env python3.6
#coding=utf-8
import sys
import pandas as pd
import numpy as np
import jieba.posseg as pseg
import re
from gensim.models import word2vec
from jieba import add_word,load_userdict
from datetime import datetime
import  codecs
load_userdict('dict.txt')
def doc_clear(s):
    if isinstance(s,str):
        r = re.sub("[A-Za-z0-9\[\`\~\!\@\#\$\^\&\*\(\)\=\|\{\}\'\:\'\[\]\<\>\/\?\~\@\#\\\&\*\%\+\（\）\=\ \、]", "", s)
    else:
        r=s
    return r

def cut_words(x,stop_words):
    #分词函数，x为字符串（待分割文本），stop_words为停用词表，list存储
    words_list=[]
    words_flag = pseg.cut(x)
    for word_temp in words_flag:
        if len(word_temp.word)>=2 and word_temp.word not in stop_words and word_temp.flag in ['v','n','a','Ag','ad','an','vg','v','vd','vn','Ng','d','i','j','l']:
            words_list.append(word_temp.word)
    return words_list


def sentence_to_words(df,str_cut,stop_words):
    #将dataframe 中每一行文本内容进行分词处理，输出结果为list。
    corpus=[]
    for i in range(len(df)):
        words_list = cut_words(df.iloc[i,:][str_cut],stop_words)
        corpus.append(words_list)
    return corpus



def load_data():
    # 读取
    df = pd.read_csv("D:\Project\Jwriter\data\Liquor\origion_data\data_liquor.csv", encoding='gbk')
    df = df.drop_duplicates()
    #停用词，读取为txt文件
    stop = set()
    fr = codecs.open('stopwords.txt', 'r', 'utf-8')
    for word in fr:
        stop.add(word.strip())
    fr.close()
    return df,stop


if __name__ == "__main__":
    start_time = datetime.now()
    # #读取数据
    data,stop_words = load_data()
    print(len(data))

    #清洗
    data['content'] = data['content'].apply(lambda x:doc_clear(x))
    #分词
    corpus = sentence_to_words(data,'content',stop_words)

    #word2vec参数
    num_features = 1000  # 词向量纬度
    min_word_count = 2  # 最小词频
    num_workers = 4  # 线程
    context = 15  # 上下文窗口大小
    downsampling = 1e-3  # 常用词采样

    print("Training model...")
    model = word2vec.Word2Vec(corpus, workers=num_workers, \
                              size=num_features, min_count=min_word_count, \
                              window=context, sample=downsampling)

    model.init_sims(replace=True)
    model_name = "word2vec_model1"
    model.save(model_name)
    # model = Word2Vec.load("D:\Model\word2vec_model")
    print(model.most_similar(u"好"))


    print(u'任务结束.....')
    print('共用时：')
    print(datetime.now()-start_time)
