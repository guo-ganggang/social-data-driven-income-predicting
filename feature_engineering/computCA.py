#! /usr/bin/python
# coding=utf-8
import codecs
import re


def computOAScores(in_file_path1,in_file_path2,out_file_path):
    with codecs.open(in_file_path1, "rb", "utf-8") as input_file1:
        for line1 in input_file1.readlines():
            oaKscores = []
            wordslist = []
            clean = ""
            for i in line1.split('\t')[1:][0].split(" "):
                wordslist.append(i.strip())
            m = 0
            while(m<6):
                with codecs.open(in_file_path2, "rb", "utf-8") as input_file2:
                    wordslist1 = {}
                    for line2 in input_file2.readlines():
                        if (line2.split(',')[0] == str(m)):
                            wordslist1[line2.split(',')[1]] = line2.split(',')[2]
                k = 0
                uidMidScore = 0.0
                uidEndScore = 0.0
                while (k < len(wordslist)):
                    if wordslist[k] in wordslist1.keys():
                        #print wordslist[k]
                        uidMidScore += float(wordslist1[wordslist[k]])
                    k += 1
                uidEndScore = uidMidScore / len(wordslist)
                oaKscores.append(uidEndScore)
                #print m,oaKscores[m]
                m += 1
            # oaKscores.append(max(oaKscores))
            # oaKscores.append(min(oaKscores))
            clean = re.sub('\[', '', str(oaKscores[0:]))
            clean = re.sub(']', '', clean)
            clean = re.sub('\s+', '', clean)
            clean = re.sub(',', '\t', clean)
            with codecs.open(out_file_path, "a+", "utf-8") as output_file:
                output_file.write(str(line1.split('\t')[0]) +'\t'+ clean +'\n')


if __name__ == '__main__':
    filePath = "D:\\incomeLevelPrediction\\db_file\\"
    inFilePath1 = filePath + "train_data_20\\train_data\\train_uid_text_incomeLevel.csv"
    inFilePath2 = filePath + "catigory_keywords\\select_catigory_keywords_1000_v2.csv"
    outFilePath = filePath +  "catigory_keywords\\catigory_keywords_1000_vec_v2.csv"

    computOAScores(inFilePath1,inFilePath2,outFilePath)










