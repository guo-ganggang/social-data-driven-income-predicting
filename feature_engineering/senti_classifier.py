#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 15/10/2016 8:52 PM
# @Author  : GUO Ganggang
# @email   : ganggangguo@csu.edu.cn
# @Site    : 
# @File    : senti_classifier.py
# @Software: PyCharm

import gensim
import numpy as np
import codecs
from gensim import utils
from os import listdir

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def similarity_matrix(fpath,opath):
    words_list = []
    with codecs.open(fpath, "rb", "utf-8") as infile:
        for line in infile:
            temp = line.strip().split(" ")
            if len(temp) != 301:
                print line
                continue
            char = utils.to_unicode(temp[0])
            if u'\u4e00' <= char <= u'\u9fff':
                words_list.append(temp[0])
    print len(words_list)

    model = gensim.models.Word2Vec.load_word2vec_format(fpath, binary=False)
    list_list = []
    with codecs.open(opath, "a+", "utf-8") as outfile:
        title = ','.join(words_list)
        outfile.write('title' + ',' + title + '\n')
        for i in range(len(words_list)):
            similarity_list = []
            for j in range(len(words_list)):
                #print words_list[i],words_list[j]
                similarityResults = round(model.similarity(utils.to_unicode(words_list[i]), utils.to_unicode(words_list[j])),4)
                similarity_list.append(similarityResults)
            #print len(similarity_list)
            value = ','.join(str(v) for v in similarity_list)
            outfile.write(words_list[i] + ',' + value + '\n')
            list_list.append(similarity_list)
    Vectors = np.matrix(list_list)
    #print Vectors

def mergeTrainData(opath,ipath_2,fileNameFeature):
    senti_number_dir = {'neutral': 0, 'angry': 1, 'dislike': 2, 'joyful': 3, 'sad': 4, 'scared': 5, 'surprized': 6}
    fileNameLabels = [f for f in listdir(ipath_2) if f.endswith(fileNameFeature)]
    with open(opath, 'a+') as output_file:
        output_file.write('text' + '\t' + 'income_level_index' + '\n')
        for fileName in fileNameLabels:
            income_level_index = 0
            for key in senti_number_dir.keys():
                if key in fileName:
                    income_level_index = senti_number_dir[key]
            with codecs.open(ipath_2 + fileName, "rb", "utf-8") as infile:
                for line in infile:
                    temp = line.strip().split(" ")
                    text = " ".join(temp[0:])
                    output_file.write(text + '\t' + str(income_level_index) + '\n')

def buildVector(ipath_1,ipath_2,opath,fileNameFeature):
    model = gensim.models.Word2Vec.load_word2vec_format(ipath_1, binary=False)
    senti_number_dir = {'neutral':0,'angry':1,'dislike':2,'joyful':3,'sad':4,'scared':5,'surprized':6}
    fileNameLabels = [f for f in listdir(ipath_2) if f.endswith(fileNameFeature)]
    with open(opath + 'senti_classifier_train_data_minCount3_vec.csv', 'a+') as output_file:
        output_file.write('text_vec' + '\t' + 'income_level_index' + '\n')
        for fileName in fileNameLabels:
            income_level_index = 0
            for key in senti_number_dir.keys():
                if  key in fileName:
                    income_level_index = senti_number_dir[key]
            with codecs.open(ipath_2 + fileName, "rb", "utf-8") as infile:
                for line in infile:
                    temp = line.strip().split(" ")
                    someWord_vec_raw = []
                    for i in range(400):
                        someWord_vec_raw.append(0.0)
                    for j in range(len(temp)):
                        try:
                            someWord_vec_new = model[utils.to_unicode(temp[j])]
                        except:
                            print temp[j]
                            continue
                        someWords_vec = list(map(lambda x:x[0]+x[1], zip(someWord_vec_raw,someWord_vec_new)))
                        someWord_vec_raw = someWords_vec
                    #print someWord_vec_raw

                    text_vec = " ".join(str(v) for v in someWord_vec_raw)
                    output_file.write(text_vec + '\t' + str(income_level_index) +'\n')


if __name__ == "__main__":

    file_path = "D:\incomeLevelPrediction\db_file\senti_6class\senti_train_data_7class_seg\\"
    opath = file_path + "all_senti_class_train_data.csv"
    fileNameFeature = '.csv'
    mergeTrainData(opath, file_path, fileNameFeature)

    fpath = "D:\\incomeLevelPrediction\\db_file\\"
    ipath_1 = fpath + "w2v_model_clean_Kwords\\3words_w2v_sg0_size400_minCount2_sample1e-4\\all_uid_text_3words.vector"
    ipath_2 = fpath + "senti_6class\\senti_train_data_7class_seg\\"
    opath = fpath + "senti_6class\\senti_train_data_7class_vec\\"
    fileNameFeature = "_seg_clean_3words.csv"
    buildVector(ipath_1,ipath_2,opath,fileNameFeature)


    # model = gensim.models.Word2Vec.load_word2vec_format(ipath_1, binary=False)
    # results = model.most_similar(u"惊讶",topn=10)
    #someWord_vec = model[u"[哼]"]
    # print someWord_vec
    # for e in results:
    #     print e[0], e[1]
    # similarityResults = model.similarity(u'锻炼',u'健康')
    # print similarityResults
    #spectral = spectral_clustering(model, n_cluster  