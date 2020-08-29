#! /usr/bin/python
# coding=utf-8
import MySQLdb
import imp
import codecs
import sys
import time
import datetime as dt

reload(sys)
sys.setdefaultencoding('utf-8')
import re

#according to the uid obtain the time
cursorObj = imp.load_source("dbCursor", "DB/WeiboDBConnection.py")
def getDemographicData(inFilePath,output_path):
        start_time = dt.datetime.now()
        uid_income = []
        with codecs.open(inFilePath, "rb", "utf-8") as input_file:
            for line in input_file.readlines():
                temp = line.strip().split('\t')
                uid_income.append(temp[0])

        with open(output_path, 'w') as output_file:
            flag = 0
            MAX_SUM = 0
            for i in range(len(uid_income)):
                uid = uid_income[i]
                #print uid
                sql = "select uid,followee_num,follower_num,weibo_num,gender,created_at from sina_weibo.weibo_users where uid = '%s'" % uid
                try:
                    con, cursor = cursorObj.getDBConnection()
                    cursor.execute(sql)
                    person_rows = cursor.fetchall()
                    for person_row in person_rows:
                        if len(person_row) != 6:
                            flag += 1
                            continue
                        #print len(person_row)
                        end_date = dt.datetime.strptime(str(person_row[5]), "%Y-%m-%d %H:%M:%S")
                        duration_registration = (start_time - end_date).days
                        # if temp[1] == 'None':
                        #     temp[1] = '0'
                        # if temp[2] == 'None':
                        #     temp[2] = '0'
                        # if temp[3] == 'None':
                        #     temp[3] = '0'
                        # sum = int(person_row[1]) + int(person_row[2]) + int(person_row[3])
                        # if sum > MAX_SUM:
                        #     MAX_SUM = sum
                        if person_row[4] == 'F':
                            gender = 0
                        else:
                            gender = 1
                        output_file.write(str(person_row[0]) +  '\t' + str(person_row[1]) + '\t' + str(person_row[2]) \
                                          + '\t' + str(person_row[3]) + '\t' + str(gender) \
                                           + '\t' + str(duration_registration) + '\n')
                        #+ '\t' + str(sum)
                    cursor.close()
                    con.close()
                except MySQLdb.Error as e:
                    print e
            print flag,MAX_SUM

if __name__ == "__main__":
    filePath = "D:\\incomeLevelPrediction\\db_file\\"
    inPath = filePath + "train_data_20\\train_data\\train_uid_text_incomeLevel.csv"
    outPath = filePath + "demographic\\get_train_data_demographic.csv"
    getDemographicData(inPath,outPath)

