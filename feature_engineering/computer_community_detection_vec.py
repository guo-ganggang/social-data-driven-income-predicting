#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 23/10/2016 9:39 AM
# @Author  : GUO Ganggang
# @email   : ganggangguo@csu.edu.cn
# @Site    : 
# @File    : computer_community_detection_vec.py
# @Software: PyCharm

import codecs
from itertools import islice

def community_detection_feature(inputPath_1,inputPath_2,inputPath_3,outputPath):

    # 获取训练集所有uid
    uid_incomeLevel_trainData = set()
    with codecs.open(inputPath_1, "rb", "utf-8") as inputfile:
        for line in islice(inputfile.readlines(),1,None):
            temp = line.strip().split('\t')
            uid_incomeLevel_trainData.add(temp[0])
    print len(uid_incomeLevel_trainData)

    # 获取大V账号的uid
    uid_bigV_uid = set()
    with codecs.open(inputPath_2, "rb", "utf-8") as inputfile:
        for line in inputfile.readlines():
            temp = line.strip()
            uid_bigV_uid.add(temp)
    print len(uid_bigV_uid)

    # 匹配并写入文件中
    with open(outputPath, 'w') as output_file:
        with codecs.open(inputPath_3, "rb", "utf-8") as inputfile:
            for line in inputfile.readlines():
                temp = line.strip().split('\t')
                if (temp[0] in uid_incomeLevel_trainData) or (temp[0] in uid_bigV_uid):
                    output_file.write(temp[0] + '\t' + temp[1] + '\n')
                    continue
                elif (temp[1] in uid_incomeLevel_trainData) or (temp[1] in uid_bigV_uid):
                    output_file.write(temp[0] + '\t' + temp[1] + '\n')
                    continue
                else:
                    continue

def cleanData(inputPath_2,outputPath):
    with open(outputPath, 'w') as output_file:
        with codecs.open(inputPath_2, "rb", "utf-8") as inputfile:
            for line in inputfile.readlines():
                temp = line.strip().split('\t')
                if ':' not in temp[1]:
                    output_file.write(temp[0] + '\t' + temp[1] + '\n')
                else:
                    print temp[1]


def noDirection(inPath,outputPath):
    uid_fee_fer_p = set()
    uid_fee_fer_n = set()
    with codecs.open(inPath, "rb", "utf-8") as input_file_1:
        for line_1 in input_file_1:
            temp = line_1.strip().split('\t')
            uid_fee_fer_p.add((temp[0], temp[1]))
            uid_fee_fer_n.add((temp[1], temp[0]))

    with open(outputPath, 'w') as output_file:
        for turple in uid_fee_fer_p:
            if tuple in uid_fee_fer_n:
                continue
            output_file.write(turple[0] + '\t' + turple[1] + '\n')


    # print len(uid_fee_fer_all)


def compCommDeteFeatVec(inputPath_1,inputPath_2,outputPath):
    # 获取训练集所有uid
    uid_incomeLevel_trainData = set()
    with codecs.open(inputPath_1, "rb", "utf-8") as inputfile:
        for line in islice(inputfile.readlines(),1,None):
            temp = line.strip().split('\t')
            uid_incomeLevel_trainData.add(temp[0])
    print len(uid_incomeLevel_trainData)

    # 获取训练集中每个uid的所属的社区号
    matchTrainUIDcommValue = {}
    staticCommValue = set()
    with codecs.open(inputPath_2, "rb", "utf-8") as inputfile:
        for line in inputfile.readlines():
            temp = line.strip().split(' ')
            if temp[0] in uid_incomeLevel_trainData:
                matchTrainUIDcommValue[temp[0]] = temp[1]
                staticCommValue.add(temp[1])
    vecLong = len(staticCommValue)
    print len(matchTrainUIDcommValue),vecLong

    for match in uid_incomeLevel_trainData:
        if match in matchTrainUIDcommValue.keys():
            continue
        else:
            matchTrainUIDcommValue[match] = '94'

    # 映射到向量
    vecMake = {}
    i = 0
    for value in staticCommValue:
        listVec = []
        for j in range(vecLong):
            listVec.append('0')
        listVec[i] = '1'
        vecMake[value] = listVec
        i += 1

    # 写入文件
    with open(outputPath, 'w') as output_file:
        for uid_train in matchTrainUIDcommValue.keys():
            temp_key = matchTrainUIDcommValue[uid_train]
            vec = vecMake[temp_key]
            temp_vec = " ".join(vec[0:])
            output_file.write(uid_train + '\t' + temp_vec + '\n')

if __name__ == "__main__":
    # filePath = 'D:\\incomeLevelPrediction\\db_file\\'
    # inputPath_1 = filePath + 'train_data_20\\train_data_merge\\train_uid_text_incomeLevel.csv'
    # inputPath_2 = filePath + 'community_finding\\data\\bigVaccount_5000000.txt'
    # inputPath_3 = filePath + 'community_finding\\data\\weibo_fee_fer_dedup.csv'
    # outputPath = filePath + 'community_finding\\data\\weibo_fee_fer_dedup_bigV_5000000.csv'
    # community_detection_feature(inputPath_1, inputPath_2,inputPath_3,outputPath)

    # cleanData(inputPath_2, outputPath)

    # inPath = filePath + 'community_finding\\data\\weibo_fee_fer_dedup_bigV_50000.csv'
    # outputPath = filePath + 'community_finding\\data\\weibo_fee_fer_dedup_bigV_50000_nodirection.csv'
    # noDirection(inPath, outputPath)

    filePath = 'D:\\incomeLevelPrediction\\db_file\\'
    inputPath_1 = filePath + 'train_data_20\\train_data_merge\\train_uid_text_incomeLevel.csv'
    inputPath_2 = filePath + 'community_finding\\GANXiS_v3.0.2\\uidFeeFer_100000\\' \
                             'SLPAw_weibo_fee_fer_dedup_bigV_100000_run1_r0.5_v3_T100.icpm.node-com.txt'
    outputPath = filePath + 'prediction_train_data_allFeatures\\community_features\\commDeteFeat_100000_vec.csv'
    compCommDeteFeatVec(inputPath_1, inputPath_2, outputPath)

