#! /usr/bin/python
# coding=utf-8
import MySQLdb
import imp
import codecs
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

cursorObj = imp.load_source("dbCursor", "DB/WeiboDBConnection.py")
def isRetweet(inFilePath,output_path):
    uid_income = {}
    with codecs.open(inFilePath, "rb", "utf-8") as input_file:
        for line in input_file.readlines():
            temp = line.strip().split('\t')
            uid_income[temp[0]] = temp[2]
    print len(uid_income)
    try:
        # uid_mid_retweet = {}
        # for key in uid_income.keys():
        #     uid = str(key).strip()
        with open(output_path + 'all_retweet_uid.csv', 'w') as output_file:
            sql = "select uid from sina_weibo.weibo_timelines where text like '转发微博%'"
            con, cursor = cursorObj.getDBConnection()
            cursor.execute(sql)
            rows = cursor.fetchall()
            for row in rows:
                uid_row = str(row[0]).strip()
                output_file.write(str(uid_row) + '\n')
                # uid_mid_retweet[uid_row] = uid_mid_retweet.get(uid_row, 0) + 1
            # print len(uid_mid_retweet)

        #uid_mid_all = {}
        with open(output_path + 'trainData_retweet_uid.csv', 'w') as output_file:
            for key in uid_income.keys():
                uid = str(key).strip()
                sql = "select uid from sina_weibo.weibo_timelines where uid = '%s'" % uid
                cursor.execute(sql)
                person_rows = cursor.fetchall()
                for person_row in person_rows:
                    uid_person_row = str(person_row[0]).strip()
                    output_file.write(str(uid_person_row) + '\n')
                    #uid_mid_all[uid_person_row] = uid_mid_all.get(uid_person_row,0) + 1
            #print len(uid_mid_all)

        cursor.close()
        con.close()
    except MySQLdb.Error as e:
        print e

def compute_retweet_ratio(inFilePath,output_path):
    all_uid_mid_retweet = {}
    with codecs.open(inFilePath + 'all_retweet_uid.csv', "rb", "utf-8") as input_file:
        for line in input_file.readlines():
            temp = line.strip()
            all_uid_mid_retweet[temp] = all_uid_mid_retweet.get(temp, 0) + 1
    print len(all_uid_mid_retweet)

    train_uid_mid_all = {}
    with codecs.open(inFilePath + 'trainData_retweet_uid.csv', "rb", "utf-8") as input_file:
        for line in input_file.readlines():
            temp = line.strip()
            train_uid_mid_all[temp] = train_uid_mid_all.get(temp, 0) + 1
    print len(train_uid_mid_all)

    with open(output_path, 'w') as output_file:
        for key in train_uid_mid_all.keys():
            #print key
            if key in all_uid_mid_retweet.keys():
                retweet_ratio = round((float(all_uid_mid_retweet[key]+1)/ float(train_uid_mid_all[key]+1)),4)
            else:
                #print train_uid_mid_all[key]
                retweet_ratio = round((float(1)/ float(train_uid_mid_all[key]+1)),4)
                #retweet_ratio = 0.0
            output_file.write(str(key) + '\t' + str(retweet_ratio) + '\n')


if __name__ == "__main__":
    filePath = "D:\\incomeLevelPrediction\\db_file\\train_data_20\\behaver_demographic_data\\"
    #inPath = filePath + "train_data\\train_uid_text_incomeLevel.csv"
    #outPath = filePath + "behaver_demographic_data\\"
    outPath = filePath + "behavioral_retweet_ratio_value.csv"
    # isRetweet(inPath,outPath)
    compute_retweet_ratio(filePath,outPath)
