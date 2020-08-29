#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/10/2016 12:44 AM
# @Author  : GUO Ganggang
# @email   : ganggangguo@csu.edu.cn
# @Site    : 
# @File    : dedup_data.py
# @Software: PyCharm
import codecs

def dedup(inFilePath,outFilePath):
    values = set()
    with codecs.open(outFilePath, "w", "utf-8") as output_file:
        with codecs.open(inFilePath, "rb", "utf-8") as inputStockCode:
            for line in inputStockCode:
                temp = line.strip()
                if temp in values:
                    continue
                else:
                    output_file.write(temp + '\n')
                    values.add(temp)

def dealwithData(inFilePath,outFilePath):
    with codecs.open(outFilePath, "w", "utf-8") as output_file:
        output_file.write('uid' + '\t' + 'text' + '\n')
        with codecs.open(inFilePath, "rb", "utf-8") as inputStockCode:
            for line in inputStockCode:
                temp = line.strip().split(' ')
                uid = temp[0]
                text = " ".join(temp[1:])
                output_file.write(uid + '\t'+ text + '\n')

def del_data(inFilePath,outFilePath):
    angry_list = 0
    scared_list = 0
    joyful_list = 0
    sad_list = 0
    dislike_list = 0
    surprized_list = 0
    with codecs.open(outFilePath, "w", "utf-8") as output_file:
        with codecs.open(inFilePath, "rb", "utf-8") as inputStockCode:
            for line in inputStockCode:
                temp_line = line.strip()
                temp = temp_line.split(',')
                if temp[2] == '1':
                    angry_list += 1
                    # if angry_list % 8 == 0:
                    #     continue
                elif temp[2] == '2':
                    joyful_list += 1
                    # if joyful_list % 2 == 0:
                    #     continue
                    # temp[2] = '1'
                elif temp[2] == '3':
                    sad_list += 1
                    # if sad_list % 2 == 0:
                    #     continue
                    # temp[2] = '2'
                elif temp[2] == '4':
                    scared_list += 1
                    # temp[2] = '3'
                # elif temp[2] == '5':
                #     scared_list += 1
                    # temp[2] = '4'
                # elif temp[2] == '6':
                #     surprized_list += 1
                    # temp[2] = '4'
                # elif temp[2] == '0':
                #     continue
                # text_seg, w2v_vec, income_level_flag, LDA_topic, LDA_topic_probability
                # 'angry': 1, 'scared': 2 , 'joyful': 3, 'sad': 4
                # 'neutral': 0, 'angry': 1, 'dislike': 2, 'joyful': 3, 'sad': 4, 'scared': 5, 'surprized': 6
                output_file.write(temp[0]+','+temp[1]+','+temp[2]+','+temp[3]+','+temp[4]+','+ '\n')
    #print angry_list,scared_list,joyful_list,sad_list,dislike_list,surprized_list
    print angry_list, joyful_list, sad_list, scared_list
    # v2  2361 3963 1986 1998


if __name__ == "__main__":
    # filePath = 'D:\\incomeLevelPrediction\\db_file\\all_seg_data_clean_Kwords\\'
    # inFilePath = filePath + 'merge_headerFalse_all_uid_text_clean_than20.csv'
    # outFilePath = filePath + 'merge_headerTrue_all_uid_text_clean_than20.csv'
    # senti_list = ['joyful.txt','sad.txt','angry.txt','surprized.txt','scared.txt','dislike.txt']
    # for i in range(len(senti_list)):
    #     inFilePath = filePath + senti_list[i]
    #     outFilePath = filePath + 'senti_6class\\' + senti_list[i]
    #     dedup(inFilePath,outFilePath)

    # dealwithData(inFilePath,outFilePath)

    filePath = 'D:\\incomeLevelPrediction\\db_file\\senti_6class\\word2vec_ldaFeature\\'
    inFilePath = filePath + 'senti_train_data_word2vec-2-400_lda_feature_4class_v4.csv'
    outFilePath = filePath + 'senti_train_data_word2vec-2-400_lda_feature_4class_new.csv'
    del_data(inFilePath,outFilePath)