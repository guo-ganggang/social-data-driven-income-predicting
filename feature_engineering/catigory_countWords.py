__author__ = 'GGG'
# -*- coding: UTF-8 -*-

import matplotlib.pyplot as pyplot
import matplotlib
import codecs
import os
import math
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def stasticWordFrequency(inFilePath,classNum,outFilePath):
    wordslist = []
    num = 0
    with codecs.open(inFilePath, "rb", "utf-8") as input_file:
        for line in input_file.readlines():
            flag = int(line.split('\t')[2])
            if(flag == classNum):
                for j in line.split('\t')[1].split(" "):
                    wordslist.append(j.strip())
                num += 1
        print '---------------' + str(classNum) + 'class contains uids: ',num
    countEveClass(wordslist,classNum,outFilePath,str(num))

def countEveClass(ClassWordslist,classNum,outFilePath,Num):
    hist = {}
    for word in ClassWordslist:
        hist[word] = hist.get(word,0) + 1

    #font = matplotlib.font_manager.FontProperties(fname='c:\\windows\\Fonts\\simsun.ttc')
    all_words_num = 0
    for key in hist.keys():
        all_words_num += hist[key]

    hist_sorted = sorted(hist.iteritems(), key=lambda d: d[1], reverse=True)
    with codecs.open(outFilePath, "a+", "utf-8") as output_file:
        for i in range(1200):
            #print hist_sorted[i][0],hist_sorted[i][1]
            tf = hist_sorted[i][1]
            logTF_logAllWords = round(math.log(tf, 4) / math.log(all_words_num,4),8)
            output_file.write(str(classNum) + "," + hist_sorted[i][0] + "," + str(hist_sorted[i][1])  + "," + Num + "," + str(logTF_logAllWords) + "\n")
    #print '-------------','show' + Num +'barChat'
    # bar_width = 0.35
    # pyplot.bar(range(60), [hist_sorted[i][1] for i in range(60)],bar_width)
    # pyplot.xticks(range(60), [hist_sorted[i][0] for i in range(60)], fontproperties=font,rotation=30)
    # pyplot.title(u"WORDS FREQUENCY" + u"by GGG",fontproperties=font)
    # pyplot.show()

if __name__ == '__main__':
    filePath = "D:\\incomeLevelPrediction\\db_file\\"
    inPath = filePath + "train_data_20\\train_data\\train_uid_text_incomeLevel.csv"
    outPath = filePath + "catigory_keywords\\catigory_keywords_1000_v2.csv"
    i = 0
    while(i<6):
        stasticWordFrequency(inPath,i,outPath)
        i += 1


