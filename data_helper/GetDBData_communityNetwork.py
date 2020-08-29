#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17/10/2016 10:27 AM
# @Author  : GUO Ganggang
# @email   : ganggangguo@csu.edu.cn
# @Site    : 
# @File    : GetDBData_communityNetwork.py
# @Software: PyCharm

import MySQLdb
import codecs
import imp

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

cursorObj = imp.load_source("dbCursor", "DB/WeiboDBConnection.py")
def obtainCommunityNetwork(inFilePath,output_path):
    uid_income = {}
    with codecs.open(inFilePath, "rb", "utf-8") as input_file:
        for line in input_file.readlines():
            temp = line.strip().split('\t')
            uid_income[temp[0]] = temp[2]
    print len(uid_income)

    with open(output_path + "uid_fee.csv", 'w') as output_file_fee:
        for key in uid_income.keys():
            uid = str(key).strip()
            sql = "select `fee_uid` from sina_weibo.weibo_followees where uid = " + uid
            try:
                con, cursor = cursorObj.getDBConnection()
                cursor.execute(sql)
                rows = cursor.fetchall()
                for row in rows:
                    fee_uid = str(row[0]).strip()
                    output_file_fee.write(uid + '\t' + fee_uid + '\n')

                cursor.close()
                con.close()
            except MySQLdb.Error as e:
                print e

    with open(output_path  + "uid_fer.csv", 'w') as output_file_fer:
        for key in uid_income.keys():
            uid = str(key).strip()
            sql = "select `fer_uid` from sina_weibo.weibo_followers where uid = " + uid
            try:
                con, cursor = cursorObj.getDBConnection()
                cursor.execute(sql)
                rows = cursor.fetchall()
                for row in rows:
                    fer_uid = str(row[0]).strip()
                    output_file_fer.write(uid + '\t' + fer_uid + '\n')

                cursor.close()
                con.close()
            except MySQLdb.Error as e:
                print e

def dedupData(inPath,outPath):
    # 无重复读入
    total_double_points = set()
    with codecs.open(inPath, "rb", "utf-8") as input_file:
        # flag = 0
        for line in input_file:
            # temp_list = []
            temp_turple = ()
            temp = line.strip().split(',')
            # temp_list.append(temp[0])
            # temp_list.append(temp[1])
            temp_turple = (temp[0],temp[1])
            total_double_points.add(temp_turple)
            # flag += 1
    print len(total_double_points)

    # 无向无重复读入
    # total_double_points_dedup = set()
    # for turple_temp in total_double_points:
    #     swaping_element = (turple_temp[1],temp_turple[0])
    #     total_double_points.add(swaping_element)
    # print len(total_double_points)

    # 写入文件
    with open(outPath, 'w') as output_file:
        for every_turple in total_double_points:
            # if (':' in every_turple[0]) or (':' in every_turple[1]):
            #     continue
            # else:
            #     output_file.write(every_turple[0]+'\t'+every_turple[1] + '\n')
            output_file.write(every_turple[0] + '\t' + every_turple[1] + '\n')

# uid 关注对象和被关注对象是同一uid
def frient_community_data(inPath_1,inPath_2,outPath):
    uid_fee = set()
    with codecs.open(inPath_1, "rb", "utf-8") as input_file_1:
        for line_1 in input_file_1:
            temp = line_1.strip().split('	')
            uid_fee.add((temp[0],temp[1]))
    print len(uid_fee)

    uid_fer = set()
    with codecs.open(inPath_2, "rb", "utf-8") as input_file_2:
        for line_2 in input_file_2:
            temp = line_2.strip().split('	')
            uid_fer.add((temp[0],temp[1]))
    print len(uid_fer)

    uid_friend = set()
    for turple_fer in uid_fer:
        if (turple_fer in uid_fee) or ((turple_fer[1],turple_fer[0]) in uid_fee):
            uid_friend.add(turple_fer)
    print len(uid_friend)

    with open(outPath, 'w') as output_file:
        for turple_fri in uid_friend:
            output_file.write(turple_fri[0] + '\t' +turple_fri[1] + '\n')

# 获得互相关注的关系对

def followEachOther(inPath,outPath):
    uid_fee_fer_all = set()
    with codecs.open(inPath, "rb", "utf-8") as input_file_1:
        for line_1 in input_file_1:
            temp = line_1.strip().split('\t')
            uid_fee_fer_all.add((temp[0], temp[1]))
    print len(uid_fee_fer_all)

    uid_fee_fer_part = set()
    with open(outPath, 'w') as output_file:
        for turple_fri in uid_fee_fer_all:
            uid_fee_fer_part.add(turple_fri)
            if (turple_fri[1],turple_fri[0]) in uid_fee_fer_all:#) and ((turple_fri[1],turple_fri[0]) not in uid_fee_fer_part):
                output_file.write(turple_fri[0] + '\t' + turple_fri[1] + '\n')



if __name__ == "__main__":
    # filePath = "D:\\incomeLevelPrediction\\db_file\\"
    # inPath = filePath + "train_data_20\\train_data_merge\\train_uid_text_incomeLevel.csv"
    # outPath = filePath + "community_finding\\"
    # obtainCommunityNetwork(inPath,outPath)

    # filePath = "D:\\incomeLevelPrediction\\db_file\\community_finding\\data\\"
    # inPath = filePath + "weibo_fee_fer_raw.csv"
    # outPath = filePath + "weibo_fee_fer_dedup.csv"
    # dedupData(inPath,outPath)

    # filePath = "D:\\incomeLevelPrediction\\db_file\\"
    # inPath_1 = filePath + "community_finding\\uid_fee.csv"
    # inPath_2 = filePath + "community_finding\\uid_fer.csv"
    # outPath = filePath + "community_finding\\train_uid_friend.txt"
    # frient_community_data(inPath_1,inPath_2,outPath)

    filePath = "D:\\incomeLevelPrediction\\db_file\\"
    inPath = filePath + "community_finding\\GANXiS_v3.0.2\\weibo_fee_fer_dedup.csv"
    outPath = filePath + "community_finding\\GANXiS_v3.0.2\\weibo_fee_fer_dedup_friend.csv"
    followEachOther(inPath, outPath)