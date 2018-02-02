#! /usr/bin/env python3.6
#coding=utf-8
import sys
import pandas as pd
import numpy as np
import jieba.posseg as pseg
import re
from gensim.models import word2vec
from jieba import add_word
from datetime import datetime

def doc_clear(s):
    if isinstance(s,str):
        r = re.sub("[A-Za-z0-9\[\`\~\!\@\#\$\^\&\*\(\)\=\|\{\}\'\:\'\[\]\<\>\/\?\~\@\#\\\&\*\%\+\（\）\=\ \、]", "", s)
    else:
        r=s
    return r

def cut_words(df,str_cut,stop_words):
    words_list=[]
    for i in range(len(df)):
        if isinstance(df.iloc[i,:][str_cut],str):
            words_flag = pseg.cut(df.iloc[i,:][str_cut])
            temp_list = []
            for word_temp in words_flag :
                if word_temp.word not in stop_words:
                # if word_temp.flag in ['v','n','a','Ag','ad','an','vg','v','vd','vn','y','Ng','nr','nt']:
                    temp_list.append(word_temp.word)
        else:
            temp_list=df.iloc[i,:][str_cut]
        words_list.append(temp_list)
    return words_list


def add_words(words):
    for word in words:
        add_word(word, freq=1000000, tag=None)

def similarity_words_find(model,sentiment_words,alpha):
    # model's vocabulary.
    index2word_set = set(model.wv.index2word)
    similar_words = pd.DataFrame()
    for word in sentiment_words:
        if word in index2word_set:
            temp = model.wv.most_similar(word)
            word_sim_list = []
            for sim_word in temp:
                if sim_word[0] not in sentiment_words and sim_word[1]> alpha :
                    word_sim_list.append(sim_word)
            word_sim_list = pd.DataFrame(word_sim_list,columns=['sim_word','sim'])
            word_sim_list['word']=word
            similar_words = pd.concat([similar_words,word_sim_list])
    return similar_words

def load_data():
    # 读取
    # df = pd.read_csv("D:\data\sku_data_3_cates.csv.gz", sep="\001",nrows=100)
    df = pd.read_csv("D:\Project\Jwriter\data\Liquor\origion_data\data_liquor.csv", encoding='gbk')
    df = df.drop_duplicates()
    stop_words = pd.read_csv("D:\Project\Jwriter\dictionary\stopwords.csv",encoding='gbk')
    dict = pd.read_csv("D:\Project\Jwriter\dictionary\word_dict_liquor.csv", encoding='gbk')
    dict = dict.drop_duplicates()
    return df,stop_words,dict


if __name__ == "__main__":
    start_time = datetime.now()
    # #读取数据
    data,stop_words,dict = load_data()
    print(len(data))
    dict_list = list(dict['word'])
    dict_list = dict_list + list(dict['word2'])
    dict_list = [w for w in dict_list if isinstance(w, str)]
    add_words(dict_list)

    #清洗
    data['content'] = data['content'].apply(lambda x:doc_clear(x))
    #分词
    corpus = cut_words(data,'content',list(stop_words['word']))

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
    model_name = "D:\Project\Jwriter\data\Liquor\word2vec_model1"
    model.save(model_name)
    # model = Word2Vec.load("D:\Model\word2vec_model")
    print(model.most_similar(u"好"))

    sim_result = similarity_words_find(model,dict_list,0.7)
    sim_result.to_csv("D:\Project\Jwriter\data\Liquor\sim_result.csv")
    print(sim_result)


    print(u'任务结束.....')
    print('共用时：')
    print(datetime.now()-start_time)
