#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/10/2016 8:09 PM
# @Author  : GUO Ganggang
# @email   : ganggangguo@csu.edu.cn
# @Site    :
# @File    : del_less_k_words.py
# @Software: PyCharm

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import codecs

def del_less_Kwords(inFilePath,outFilePath,k,clean_list):
    values = set()
    with codecs.open(outFilePath, "w", "utf-8") as output_file:
        with codecs.open(inFilePath, "rb", "utf-8") as inputStockCode:
            for line in inputStockCode:
                temp = line.strip().split('\t')
                date  = temp[0]
                text = temp[1]
                temp2 = text.strip().split(' ')
                # temp2 = line.strip().split(' ')

                clean_temp = []
                for i in range(len(temp2)):
                    flag = '0'
                    for string in clean_list:
                        if string in temp2[i]:
                            #print temp[i]
                            flag = '1'
                        else:
                            continue
                    if flag == '1':
                        continue
                    else:
                        clean_temp.append(temp2[i])
                lenth = len(clean_temp)
                vec = " ".join(clean_temp[0:])
                #print lenth
                if lenth >= k:
                    if vec in values:
                        continue
                    else:
                        output_file.write(date + '\t' +vec + '\n') #
                        values.add(vec)
                    #output_file.write(vec + '\n')
                # elif lenth < k:
                #     print lenth,k
                #     print vec

if __name__ == "__main__":
    filePath = 'D:\\pingan_stock\\financial_news\\'
    k = 1
    clean_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10','0', "'''", "''", '．', '……', ',','﹍', \
                  '＂', '／', '〜', '•', '➊', '➋', '➌', '➍', '➎', '➏', '﹑', '´','╬╬', '＼', '﹎','﹏',\
                  '︻', '.', '%', '﹌', 'ミ','∟', '☆','…','➠', '➕','•','a','b','c','d', 'e','f','g','h',\
                  'i', 'j', 'k', 'l', 'm', 'n','o','p','q','r','s', 't','u','v','w','x','y','z','ゞ','∷', \
                  '￡', '╬', 'з','ヽ','｀', '︱', '┢┦', '※','�','▄','┻═','┳一', '－','≦','〆']

    #for i in range(6):
    inFilePath = filePath + 'pingan_finance_news_seg.csv'
    outFilePath = filePath + 'pingan_finance_news_seg_clean_' + str(k) +'words.csv'
    del_less_Kwords(inFilePath,outFilePath,k,clean_list)
