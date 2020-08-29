#!/usr/bin/python
# -*- coding=utf-8 -*-
import gensim
from sklearn.cluster import spectral_clustering
#import pandas as pd
import numpy as np
import codecs
import sys
from gensim import utils
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

if __name__ == "__main__":
    fpath = "D:\\pingan_stock\\financial_news\\word2vec\\"
    #ipath = fpath + "3words_w2v_sg0_size800_minCount5_sample1e-4\all_uid_text_3words.vector"
    ipath = fpath + "sg0_s200_w3_m2_3words.vector"
    # opath = "similarity_vector_v2.csv"
    model = gensim.models.Word2Vec.load_word2vec_format(ipath, binary=False)
    results = model.most_similar(u"上涨",topn=100)
    for e in results:
        print e[0], e[1]
    print model[u"上涨"]
    # similarityResults = model.similarity(u'锻炼',u'健康')
    # print similarityResults
    #spectral = spectral_clustering(model, n_cluster