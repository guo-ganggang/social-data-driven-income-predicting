import os
import MySQLdb
# how to use
import imp
import tool
import codecs
# utility = imp.load_source('utility',os.path.abspath('../utility.py'))
# utility.getFullPath('abc.py')
import sys
import csv
import string
import os
import re
import pandas as pd
import matplotlib.pyplot as plt

reload(sys)
sys.setdefaultencoding('utf-8')
import re

cursorObj = imp.load_source("dbCursor", "DB/WeiboDBConnection.py")
def getWeibo(output_path):

    sql = "select distinct uid from sina_weibo.weibo_timelines where uid is not null"
    try:
        print sql
        con, cursor = cursorObj.getDBConnection()
        cursor.execute(sql)
        rows = cursor.fetchall()
        print "all users:",cursor.rowcount

        #index = 0
        no_weibo = 0
        with open(output_path, 'w') as output_file:
            for row in rows:
                uid = str(row[0]).strip()
                sql = "select uid,text from sina_weibo.weibo_timelines where uid = '%s' and text is not null" % uid
                cursor.execute(sql)
                person_rows = cursor.fetchall()
                num_weibo = len(person_rows)
                #print num_weibo
                if num_weibo >= 20:
                    output_file.write(uid + "\t")
                    for person_row in person_rows:
                        if not person_row[1]:
                            continue
                        output_file.write(re.sub('\s', ' ', str(person_row[1])) + " ")
                    output_file.write("\n")
                else:
                    #print num_weibo
                    no_weibo += 1
                #         index += 1
                # print "no_weibo:" ,no_weibo
                # print "all users" , index
        cursor.close()
        con.close()
        print "weibo text less 20's uid number:  ", no_weibo
    except MySQLdb.Error as e:
        print e

if __name__ == "__main__":
    #getAllBaonasiWeibo("D:\\dataIO\\agFiles\\input\\predictionIncomeLevelIO\\richPeopleWeibo.txt")
    #getWeibo("D:\\dataIO\\agFiles\\input\\predictionIncomeLevelIO\\richPeopleWeibo.txt")
    outputPath = "D:\\incomeLevelPrediction\\db_file\\all_raw_seg_data\\all_weibo_uid_text_20.csv"
    getWeibo(outputPath)

