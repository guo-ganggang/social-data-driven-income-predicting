#! /usr/bin/python
# coding=utf-8
from __future__ import division
import codecs
import matplotlib.pyplot as pyplot
import matplotlib

def tfswo1(in_file_path,out_file_path,TotalClass,KNum):
    CandidateKeyWord1 = {}
    catigory_flag = {}
    with codecs.open(in_file_path, "rb", "utf-8") as input_file1:
        for line1 in input_file1.readlines():
            # print type(line.split('\t')[1:]),line.split('\t')[1:]
            # print type(line.split('\t')[1:][0]),line.split('\t')[1:][0]
            temp = line1.strip().split(',')
            CandidateKeyWord1[temp[1]] = CandidateKeyWord1.get(temp[1],0) + int(temp[2]) / int(temp[3])
            catigory_flag[temp[1]] = catigory_flag.get(temp[1],0) + 1
        num = 0
        while(num < TotalClass):
            tfswo2(in_file_path,out_file_path,CandidateKeyWord1,catigory_flag,KNum,num,TotalClass)
            num += 1

def tfswo2(in_file_path,out_file_path,CandidateKeyWord,catigory_flag,KNum,n,TotalClass):
    CandidateKeyWord2 = {}
    with codecs.open(in_file_path, "rb", "utf-8") as input_file2:
        for line2 in input_file2.readlines():
            temp = line2.strip().split(',')
            if(int(temp[0]) == n):
                idf_1 = (int(temp[2]) / int(temp[3])) / CandidateKeyWord[temp[1]]
                #idf_2 = catigory_flag[temp[1]] / float(TotalClass)
                #print idf_1,idf_2
                CandidateKeyWord2[temp[1]] = float(temp[4]) * idf_1   #* idf_2

        # for key in CandidateKeyWord2.keys():
        #     print key, CandidateKeyWord2[key]
        # for key in CandidateKeyWord2.keys():
        #     output_file.write(key + "\t" + str(CandidateKeyWord2[key]) + "\n")
        #print CandidateKeyWord2.iteritems()
        selectK_sorted = sorted(CandidateKeyWord2.iteritems(), key=lambda d: d[1], reverse=True)
        with codecs.open(out_file_path, "a+", "utf-8") as output_file:
            for i in range(KNum):
                output_file.write(str(n) + "," + selectK_sorted[i][0] + "," + str(selectK_sorted[i][1])  + "\n")

        print '-------------', 'show' + str(n) + 'barChat'
        font = matplotlib.font_manager.FontProperties(fname='c:\\windows\\Fonts\\simsun.ttc')
        bar_width = 0.35
        pyplot.bar(range(60), [selectK_sorted[i][1] for i in range(60)],bar_width)
        pyplot.xticks(range(60), [selectK_sorted[i][0] for i in range(60)], fontproperties=font,rotation=30)
        pyplot.title(u"select k score" + u"by GGG",fontproperties=font)
        pyplot.show()

if __name__ == '__main__':
    filePath = "D:\\incomeLevelPrediction\\db_file\\"
    inPath = filePath + "catigory_keywords\\catigory_keywords_1200_v2.csv"
    outPath = filePath + "catigory_keywords\\select_catigory_keywords_1000_v2.csv"
    tfswo1(inPath,outPath,6,1000)