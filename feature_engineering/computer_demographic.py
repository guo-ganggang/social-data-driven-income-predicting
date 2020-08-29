#! /usr/bin/python
# coding=utf-8
import codecs
import numpy as np

def computer_demographic(inFilePath,outFilePath):
    with codecs.open(inFilePath, "rb", "utf-8") as input_file:
        uid_influence_list = []
        for line in input_file.readlines():
            sum = 0
            temp = line.strip().split('\t')
            sum = int(temp[1]) + int(temp[2]) + int(temp[3])
            uid_influence_list.append(sum)

    array_uid_influence = np.array(uid_influence_list)
    mean_uid_influence = np.mean(array_uid_influence)
    std_uid_influence = np.std(array_uid_influence)

    print mean_uid_influence, std_uid_influence

    with codecs.open(outFilePath, "w", "utf-8") as output_file:
        output_file.write('uid' + '\t' + 'uid_influence_value' + '\n')
        with codecs.open(inFilePath, "rb", "utf-8") as input_file:
            for line in input_file.readlines():
                sum = 0
                temp = line.strip().split('\t')
                sum = int(temp[1])+int(temp[2])+int(temp[3])
                uid_influence_value = round(((float(sum) - mean_uid_influence)/ float(std_uid_influence)),8)
                output_file.write(str(temp[0]) + '\t' + str(uid_influence_value) + '\n')

def computer_gender(inFilePath, outFilePath):
    with codecs.open(outFilePath, "w", "utf-8") as output_file:
        output_file.write('uid' + '\t' + 'gender' + '\n')
        with codecs.open(inFilePath, "rb", "utf-8") as input_file:
            for line in input_file.readlines():
                temp = line.strip().split('\t')
                output_file.write(str(temp[0]) + '\t' + str(temp[4]) + '\n')

def computer_regist(inFilePath, outFilePath):
    regist_list = []
    with codecs.open(inFilePath, "rb", "utf-8") as input_file:
        for line in input_file.readlines():
            temp = line.strip().split('\t')
            regist_list.append(float(temp[5]))

    array_uid_regist = np.array(regist_list)
    mean_uid_regist = np.mean(array_uid_regist)
    std_uid_regist = np.std(array_uid_regist)

    with codecs.open(outFilePath, "w", "utf-8") as output_file:
        output_file.write('uid' + '\t' + 'gender' + '\n')
        with codecs.open(inFilePath, "rb", "utf-8") as input_file:
            for line in input_file.readlines():
                temp = line.strip().split('\t')
                regist_ratio = (float(temp[5]) - mean_uid_regist) / std_uid_regist
                output_file.write(str(temp[0]) + '\t' + str(regist_ratio) + '\n')


if __name__ == "__main__":
    filePath = "D:\\incomeLevelPrediction\\db_file\\train_data_20\\behaver_demographic_data\\"
    inFilePath = filePath + "demographic_felowee_felower_weiboNum_gender_registDuration.csv"
    outFilePath1 = filePath + "demographic_uid_influence_value.csv"
    outFilePath2 = filePath + "demographic_uid_gender_value.csv"
    outFilePath3 = filePath + "demographic_uid_regist_value.csv"
    # computer_demographic(inFilePath, outFilePath1)
    # computer_gender(inFilePath, outFilePath2)
    computer_regist(inFilePath, outFilePath3)