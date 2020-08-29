#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/10/2016 5:04 PM
# @Author  : GUO Ganggang
# @email   : ganggangguo@csu.edu.cn
# @Site    : 
# @File    : doc2vec.py
# @Software: PyCharm
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import codecs
from os import listdir
import os
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from gensim import utils
from random import shuffle
import logging
import multiprocessing

log = logging.getLogger()
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)

#给模型训练需要的数据打标签
class TaggedLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
	return self.sentences

#将字典类型中的值写入文件中
def writeData(myDirPath,fileName,data_dic):
    with codecs.open(myDirPath + os.sep + fileName, "w", "utf-8") as output_file:
        for key in data_dic.keys():
            #print data_dic[key]
            output_file.write(data_dic[key] + '\n')

#划分训练集与测试集
def readData(myDirPath,fileType,rate):
    fileNameLabels = [f for f in listdir(myDirPath) if f.endswith(fileType)]
    for fileName in fileNameLabels:
        train_dic = {}
        test_dic = {}
        with open(myDirPath + os.sep + fileName, 'r') as infile:
            tempData = infile.readlines()
            length = len(tempData)
            threshod = int(round(length * rate))
            #print threshod
            for i in range(length):
                temp = tempData[i].strip().split(' ')
                uid = temp[0]
                text = " ".join(temp[1:])
                if i <= threshod:
                    train_dic[uid] = text
                else:
                    test_dic[uid] = text
        writeData(myDirPath + 'divide_dataSet', 'train_' + fileName, train_dic)
        writeData(myDirPath + 'divide_dataSet', 'test_' + fileName, test_dic)
        print fileName,length,len(train_dic),len(test_dic)

#将语料库中包含的训练集中的数据去除
def dedup_corpus(myDirPath,fileType,inFilePath,outFilePath):
    uid_set = set()
    fileNameLabels = [f for f in listdir(myDirPath) if f.endswith(fileType)]
    for fileName in fileNameLabels:
        with open(myDirPath + os.sep + fileName, 'r') as infile:
            tempData = infile.readlines()
            for line in tempData:
                temp = line.strip().split(' ')
                uid_set.add(temp[0])

    with codecs.open(outFilePath, "w", "utf-8") as output_file:
        with codecs.open(inFilePath, "rb", "utf-8") as inHandle:
            for line in inHandle:
                temp = line.strip().split(' ')
                uid = temp[0]
                text = " ".join(temp[1:])
                if uid in uid_set:
                    continue
                else:
                    output_file.write(text + '\n')

#开始训练模型
def d2vTrain():
    log.info('source load')
    sources = { \
        'test_class_one.csv' : 'TEST_CLASS_ONE', \
        'test_class_three.csv' : 'TEST_CLASS_THREE', \
        'test_class_two.csv' : 'TEST_CLASS_TWO', \
        'test_class_four.csv' : 'TEST_CLASS_FOUR', \
        'test_class_five.csv' : 'TEST_CLASS_FIVE', \
        'test_class_zero.csv' : 'TEST_CLASS_ZERO',  \
        'train_class_five.csv' : 'TRAIN_CLASS_FIVE', \
        'train_class_four.csv' : 'TRAIN_CLASS_FOUR', \
        'train_class_one.csv' :  'TRAIN_CLASS_ONE', \
        'train_class_three.csv' :  'TRAIN_CLASS_THREE', \
        'train_class_two.csv' :  'TRAIN_CLASS_TWO', \
        'train_class_zero.csv' :  'TRAIN_CLASS_ZERO', \
        'train_unsup.csv' :  'TRAIN_UNSUP'
    }
    # sources = {}
    # fileNameLabels = [f for f in listdir(myDirPath) if f.endswith(fileType)]
    # for fileName in fileNameLabels:
    #     sources[fileName] = fileName.replace(fileType,'').upper()
    #     print fileName,fileName.replace(fileType,'').upper()
    log.info('TaggedDocument')
    sentences = TaggedLineSentence(sources)

    log.info('D2V')
    model = Doc2Vec(min_count=10, window=10, size=250, sample=1e-4, negative=5, workers=multiprocessing.cpu_count())
    model.build_vocab(sentences.to_array())

    log.info('Epoch')
    for epoch in range(30):
        log.info('EPOCH: {}'.format(epoch))
        model.train(sentences.sentences_perm())

    log.info('Model Save')
    model.save('weibo_10_10_800.d2v')


if __name__ == "__main__":
    #myDirPath = 'D:\\incomeLevelPrediction\\db_file\\d2v_model_20\\divide_dataSet\\'
    # myDirPath = 'doc2vec'
    # fileType = '.csv'
    # filePath = 'D:\\incomeLevelPrediction\\db_file\\'
    # inFilePath = filePath + 'all_seg_data_clean_Kwords\\merge_headerFalse_all_uid_text_clean_than20' +fileType
    # outFilePath = filePath + 'd2v_model_20\\divide_dataSet\\train_unsup' +fileType
    # rate = 0.75
    # readData(myDirPath, fileType, rate)
    #dedup_corpus(myDirPath, fileType, inFilePath, outFilePath)
    d2vTrain()
    #d2vTrain()