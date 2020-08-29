import MySQLdb
import imp
import codecs
import sys
import time


reload(sys)
sys.setdefaultencoding('utf-8')
import re

#according to the uid obtain the time
cursorObj = imp.load_source("dbCursor", "DB/WeiboDBConnection.py")
def getBehavierData(inFilePath,output_path):
        uid_income = {}
        with codecs.open(inFilePath, "rb", "utf-8") as input_file:
            for line in input_file.readlines():
                temp = line.strip().split('\t')
                uid_income[temp[0]] = temp[2]

        with open(output_path, 'w') as output_file:
            for key in uid_income.keys():
                uid = str(key).strip()
                sql = "select `uid`,`created_at`,`from` from sina_weibo.weibo_timelines where uid = '%s' and text is not null" % uid
                try:
                    con, cursor = cursorObj.getDBConnection()
                    cursor.execute(sql)
                    person_rows = cursor.fetchall()
                    num_weibo = len(person_rows)
                    print num_weibo

                    for person_row in person_rows:
                        string = time.strptime(person_row[1], '%Y-%m-%d %H:%M')
                        hour = string[3]
                        output_file.write(person_row[0] +  '\t' + str(hour) +  '\t' + str(person_row[2]) \
                                          + '\t' + uid_income[key] + '\n')

                    cursor.close()
                    con.close()
                except MySQLdb.Error as e:
                    print e

if __name__ == "__main__":
    filePath = "D:\\incomeLevelPrediction\\db_file\\train_data_20\\"
    inPath = filePath + "train_data\\train_uid_text_incomeLevel.csv"
    outPath = filePath + "behaver_demographic_data\\behaver_time_tool.csv"
    getBehavierData(inPath,outPath)

