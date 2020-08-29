#! /usr/bin/python
# coding=utf-8
import codecs
import re

#static total hour times
uid_num_list = {}
def stockUid(inFilePath):
    with codecs.open(inFilePath, "rb", "utf-8") as input_file:
        for line in input_file.readlines():
            temp = line.strip().split('\t')
            uid_num_list[temp[0]] = uid_num_list.get(temp[0], 1) + 1
        print len(uid_num_list)

#compute the hour times vector
def staicPostTime(inFilePath,output_path):
    with open(output_path, 'w') as output_file:
        for uid in uid_num_list.keys():
            static_hour = {}
            hourVec = []
            belong_index = ''
            with codecs.open(inFilePath, "rb", "utf-8") as input_file:
                for line in input_file.readlines():
                    temp = line.strip().split('\t')
                    if temp[0] == uid:
                        #print temp[0], uid
                        static_hour[temp[1]] = static_hour.get(temp[1], 0) + 1
                        if belong_index != '':
                            continue
                        else:
                            belong_index = temp[3]
                #print len(static_hour)

            for j in range(25):
                if str(j) in static_hour.keys():
                    #print static_hour[str(j)],uid_num_list[uid]
                    ratio_hour = round((static_hour[str(j)] / float(uid_num_list[uid])),2)
                    #print 'ratio_hour ' + str(ratio_hour)
                    hourVec.append(ratio_hour)
                else:
                    hourVec.append(0.0)
            vecItem = re.sub('\[', '',str(hourVec))
            vecItem = re.sub('\]', '', str(vecItem))
            output_file.write(str(uid) +', ' + vecItem +', '+ belong_index +'\n')

if __name__ == "__main__":
    filePath = "D:\\incomeLevelPrediction\\db_file\\train_data_20\\"
    inPath = filePath + "behaver_demographic_data\\behavioral_time_tool.csv"
    outPath = filePath + "behaver_demographic_data\\behavioral_time_vec.csv"
    stockUid(inPath)
    staicPostTime(inPath,outPath)