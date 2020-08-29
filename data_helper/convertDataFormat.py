#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 16/11/2016 5:41 PM
# @Author  : GUO Ganggang
# @email   : ganggangguo@csu.edu.cn
# @Site    : 
# @File    : convertDataFormat.py
# @Software: PyCharm

import codecs

def convertDataFormat(inFilePath,outFilePath,vec_name):
    with codecs.open(outFilePath, "w", "utf-8") as output_file:
        output_file.write('uid' + '\t' + vec_name + '\n')
        with codecs.open(inFilePath, "rb", "utf-8") as inputStockCode:
            for line in inputStockCode:
                temp = line.strip().split(',')
                length = len(temp)
                print length
                uid = temp[0]
                time_vec = " ".join(temp[1:(length-1)])
                output_file.write(uid + '\t' + time_vec + '\n')


if __name__ == "__main__":
    filePath = 'D:\\incomeLevelPrediction\\db_file\\prediction_train_data_allFeatures\\behavioral_features\\'
    inFilePath = filePath + 'behavioral_tool_vec_old.csv'
    outFilePath = filePath + 'behavioral_tool_vec.csv'
    vec_name = 'tool_vec'
    convertDataFormat(inFilePath,outFilePath,vec_name)