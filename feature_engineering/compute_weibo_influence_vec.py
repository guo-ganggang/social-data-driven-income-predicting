#! /usr/bin/python
# coding=utf-8
import codecs
import re
import numpy as np

#normalized
def normalized(filePath1,filePath2):
    with codecs.open(filePath1, "rb", "utf-8") as input_file:
        weiboInfluence_list = []
        for line in input_file.readlines():
            temp = line.strip().split('\t')
            weiboInfluence_list.append(float(temp[4]))
    array_weiboInfluence = np.array(weiboInfluence_list)
    mean_weiboInfluence = np.mean(array_weiboInfluence)
    std_weiboInfluence = np.std(array_weiboInfluence)
    print mean_weiboInfluence,std_weiboInfluence

    uid_static = {}
    with open(filePath2, 'w') as output_file:
        with codecs.open(filePath1, "rb", "utf-8") as input_file:
            for line in input_file.readlines():
                temp = line.strip().split('\t')
                ratio = round(((float(temp[4]) -mean_weiboInfluence)/ float(std_weiboInfluence)),4)
                uid_static[temp[0]] = uid_static.get(temp[0],0) + 1
                output_file.write(temp[0] + '\t'+ str(ratio) +'\n')
    print len(uid_static)
    return uid_static

#all tweets per user on average influence
def averageInfluence(filePath2,filePath3):
    uid_static = normalized(filePath1,filePath2)
    with open(filePath3, 'w') as output_file:
        for key in uid_static.keys():
            sum_influence = 0.0
            avg_influence = 0.0
            with codecs.open(filePath2, "rb", "utf-8") as input_file:
                for line in input_file.readlines():
                    temp = line.strip().split('\t')
                    if key == temp[0]:
                        sum_influence += float(temp[1])
                        #print sum_influence
            avg_influence = round((sum_influence / uid_static[key]),8)
            #print avg_influence
            output_file.write(str(key) + '\t' + str(avg_influence) + '\n')


if __name__ == "__main__":
    filePath = "D:\\incomeLevelPrediction\\db_file\\train_data_20\\behaver_demographic_data\\"
    filePath1 = filePath + "behavioral_retweet_comment_favourate.csv"
    filePath2 = filePath + "behavioral_retweet_comment_favourate_normalized.csv"
    filePath3 = filePath + "behavioral_weibo_influence_normalize.csv"
    averageInfluence(filePath2, filePath3)