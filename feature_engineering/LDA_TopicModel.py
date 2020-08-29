#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 30/9/2016 8:41 PM
# @Author  : GUO Ganggang
# @email   : ganggangguo@csu.edu.cn
# @Site    :
# @File    : obtain_train_data.py
# @Software: PyCharm

import graphlab as gl
import codecs
class TopicModel:
    def trainModel(self, trained_file, train_field, model_saved_name, num_topics, num_iterations, method='auto',
                   beta=0.01, delimiter="\t"):
        train_data = gl.SFrame.read_csv(trained_file, delimiter, header=True)
        print train_data
        train_bag = gl.text_analytics.count_words(train_data[train_field])
        train_model = gl.topic_model.create(train_bag, num_topics=num_topics, num_iterations=num_iterations, beta=beta,
                                            method=method, verbose=True)
        train_model.save(model_saved_name)
        return train_model
    # def predict(self,train_model,test_file,test_field,test_result_field,test_output_file,delimiter = " "):
    #     test_data = gl.SFrame.read_csv(test_file,delimiter = delimiter,header = True)
    #     test_bag = gl.text_analytics.count_words(test_data[test_field])
    #     test_data[test_result_field] = train_model.predict(test_bag)
    #     test_data.save(test_output_file)


def dealwithData(inFilePath,outFilePath):
    with codecs.open(outFilePath, "w", "utf-8") as output_file:
        output_file.write('uid' + '\t' + 'text' + '\n')
        with codecs.open(inFilePath, "rb", "utf-8") as inHandle:
            for line in inHandle:
                temp = line.strip().split(' ')
                uid = temp[0]
                text = " ".join(temp[1:])
                output_file.write(uid + '\t' + text + '\n')

def showLDAtopicSep(input_file,output_file):
    weibo_model = gl.load_model(input_file)
    topics_words = weibo_model.get_topics(num_words=200, output_type='topic_words')
    topics_words.save(output_file)

if __name__ == "__main__":
    # dealwithData(inFilePath, outFilePath)
    # filePath = 'D:\\incomeLevelPrediction\\db_file\\'
    # #filePath = ''
    # train_file = filePath + 'all_seg_data_clean_Kwords\\merge_headerTrue_all_uid_text_clean_than20_v2.csv'
    # train_field = 'text'
    # model_saved_name = filePath + 'LDA_model_clean_Kwords\\all_uid_text_clean_than20_merge'
    # topic_model = TopicModel()
    # train_model = topic_model.trainModel(train_file,train_field,model_saved_name,num_topics = 250,num_iterations = 1500)
    # print train_model.get_topics()


    filePath = 'D:\\incomeLevelPrediction\\db_file\\'
    input_file = filePath + 'LDA_model_clean_Kwords\\all_weibo_7_LDA_model'
    output_file = filePath + 'qualitative_analysis\\show_all_weibo_7_LDA_model_topics.csv'

    showLDAtopicSep(input_file,output_file)



