#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 5/10/2016 10:36 AM
# @Author  : GUO Ganggang
# @email   : ganggangguo@csu.edu.cn
# @Site    : 
# @File    : dealwith_train_data.py
# @Software: PyCharm

import codecs
from os import listdir
from itertools import islice
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def divide(inFilePath,outFilePath,k):
    outFilePath = outFilePath + '_' + 'class' + str(k) + '.csv'
    with codecs.open(outFilePath, "w", "utf-8") as output_file:
        with codecs.open(inFilePath, "rb", "utf-8") as inputStockCode:
            for line in inputStockCode:
                temp = line.strip().split('\t')
                if len(temp) != 3:
                    continue
                else:
                    if temp[2] == str(k):
                        output_file.write(temp[0] + '\t' + temp[1] + '\n')

def merge_6_class(filepath,fileNameFeature):
    level_list = ['0','1','2','3','4','5']
    fileNameLabels = [f for f in listdir(filepath) if f.startswith(fileNameFeature)]
    with codecs.open(filepath + '\\uid_text_incomeLevel.csv', "a+", "utf-8") as output_file:
        output_file.write('uid' + '\t' + 'text_seg' + '\t' + 'income_level_flag' + '\n')
        for fileName in fileNameLabels:
            print fileName
            income_level = ''
            for i in range(len(level_list)):
                if level_list[i] in fileName:
                    income_level = level_list[i]
            with codecs.open(filepath + fileName, 'r') as infile:
                for line in infile:
                    temp = line.strip().split(' ')
                    uid = temp[0]
                    text = " ".join(temp[1:])
                    output_file.write(uid + '\t' + text + '\t' + income_level + '\n')

def sentiTrainData(inFilePath,outFilePath):
    with codecs.open(outFilePath, "w", "utf-8") as output_file:
        output_file.write('uid' + '\t' + 'catigory_key' +'\n')
        with codecs.open(inFilePath, "rb", "utf-8") as input_file:
            for line in input_file:
                temp = line.strip().split('\t')
                catigory_key = " ".join(str(v) for v in temp[1:])
                output_file.write(temp[0] + '\t' + catigory_key +'\n')

# uid,income_level,weibo_topic,weibo_topic_probability,
# WeightedAvg,variance,positiveAvg,negativeAvg,neutralAvg,
# positiveNumRate,negativeNumRate,neutralNumRate,
# senti_vec,doc2vec,catigory_key,uid_influence_value,
# gender,regist_duration,weibo_influence,
# retweet_ratio,time_vec,tool_vec,commnu_detect_vec
# "[]"
def dataTypeChangeCentralized(inFilePath,outFilePath):
    with codecs.open(outFilePath, "w", "utf-8") as output_file:

        # output_file.write('uid' + '\t' + 'catigory_key' + '\n')
        with codecs.open(inFilePath, "rb", "utf-8") as input_file:
            for line in islice(input_file, 1, None):

                temp = line.strip().split(',')
                temp[3] = '"[' + temp[3] + ']"'
                temp[12] = '"[' + temp[12] + ']"'
                temp[13] = '"[' + temp[13] + ']"'
                temp[14] = '"[' + temp[14] + ']"'
                temp[20] = '"[' + temp[20] + ']"'
                temp[21] = '"[' + temp[21] + ']"'
                temp[22] = '"[' + temp[22] + ']"'
                catigory_key = ",".join(str(v) for v in temp[1:])
                output_file.write(temp[0] + ',' + catigory_key + '\n')

def dataTypeChangeScattered(inFilePath,outFilePath):
    with codecs.open(outFilePath, "w", "utf-8") as output_file:
        flag_list = []
        for j in range(269):
            flag_list.append("flag_" + str(j))
        catigory_name = ",".join(str(v) for v in flag_list[0:])
        output_file.write('uid' + ',' + 'income_level' +',' + catigory_name + '\n')
        with codecs.open(inFilePath, "rb", "utf-8") as input_file:
            for line in islice(input_file, 1, None):
                flag = 0
                temp = line.strip().split(',')
                temp_temp = []
                temp_select = temp[0:12] + temp[14:19]
                for key in temp_select:
                    key_temp = key.strip().split(' ')
                    # print len(key_temp)
                    for i in key_temp:
                        temp_temp.append(i)
                        flag += 1
                catigory_key = ",".join(str(v) for v in temp_temp[0:])
                output_file.write(catigory_key + '\n')
                if flag != 271:
                    print flag

if __name__ == "__main__":
    filePath = 'D:\\incomeLevelPrediction\\db_file\\'
    # k = 6
    # inFilePath = filePath + 'train_data_20\\train_merge_uid_text_incomeLevel_than20.csv'
    # outFilePath = filePath + 'train_data_20\\train_merge_6class_uid_text\\train_uid_text'
    # for i in range(6):
    #     divide(inFilePath,outFilePath,i)
    # fileNameFeature = 'class_'
    # merge_6_class(filePath, fileNameFeature)
    # inFilePath =filePath + 'prediction_train_data_allFeatures\\behavioral_features\\behavioral_time_vec_old'
    # outFilePath = filePath + 'prediction_train_data_allFeatures\\ugc_features\\ugc_ck_vec.csv'
    # sentiTrainData(inFilePath, outFilePath)


    inFilePath = filePath + 'sklearn_prediction_result\\prediction_train_data_allFeatures_merge.csv'
    outFilePath = filePath + 'sklearn_prediction_result\\prediction_train_data_allFeatures_merge_scattered.csv'
    # outFilePath = filePath + 'sklearn_prediction_result\\prediction_train_data_allFeatures_merge_centralized.csv'
    dataTypeChangeScattered(inFilePath,outFilePath)