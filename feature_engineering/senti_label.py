#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 10/10/2016 3:30 PM
# @Author  : GUO Ganggang
# @email   : ganggangguo@csu.edu.cn
# @Site    : 
# @File    : senti_label.py
# @Software: PyCharm

import codecs
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import re


def dealwithData(inFilePath,outFilePath,classLabel):
    with codecs.open(outFilePath, "w", "utf-8") as output_file:
        #output_file.write('uid' + '\t' + 'text' + '\n')
        with codecs.open(inFilePath, "rb", "utf-8") as inHandle:
            for line in inHandle:
                temp = line.strip().split('	')
                length = len(temp)
                label = temp[length-1]
                old_label = length-2
                text = "	".join(temp[0:old_label])
                if label == classLabel:
                    output_file.write(text + '\n')


def train_data_dealwith(clean_list,inFilePath,outFilePath):
    with codecs.open(outFilePath, "w", "utf-8") as outHandle:
        with codecs.open(inFilePath, "rb", "utf-8") as inHandle:
            for line in inHandle:
                flag = 0
                temp = line.strip().split(' ')
                vec = " ".join(temp[0:])
                for i in range(len(temp)):
                    if temp[i] in clean_list:
                        flag = 1
                if flag == 0:
                    outHandle.write(vec + '\n')


if __name__ == "__main__":
    filePath = "D:\\incomeLevelPrediction\\db_file\\senti_6class\\senti_train_data_7class_seg\\"
    clean_list = [ \
        '无聊','report','upload','以泪洗面','[干杯]','发红包','装腔作势','家纺','热线','微刊','大哭', '妈蛋','群号','中枪', \
        '泪花', '刷机', '大酬宾', '[挖鼻屎]', '[生病]', '嘿嘿', '[鄙视]', '[左哼哼]', '[悲伤]', '[衰]', '[晕]', \
        '[困]', '', '促销价', '美甲', '哭闹', '沙发', '恶搞', '豆瓣', '测试', '美容', '包邮','入侵','[吐]','[失望]', \
        '[抓狂]', '更多', '分享', '[好囧]', '功效', '全网', '户型','关注', '视频', '好评', '正品','一口价','推荐','新款','旗舰' \
        ]
    inFilePath = filePath + "neutral_seg_clean_3words.csv"
    outFilePath = filePath + "neutral_seg_clean_3words_new.csv"
    # classLabel = ['angry','joyful','neutral','sad','scared','dislike','surprized']
    # inFilePath = FilePath + "need_label_weibo.txt"
    # for i in range(len(classLabel)):
    #     outFilePath = FilePath + 'need_labeled_data\\' + classLabel[i] + '_neededLabel.csv'
    #     dealwithData(inFilePath,outFilePath,classLabel[i])
    train_data_dealwith(clean_list,inFilePath,outFilePath)