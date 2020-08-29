#! /usr/bin/python
# coding=utf-8
import codecs
import re

#static total the first 35 which are sorted by
postTool_35 = []
def postTools(inFilePath):
    tool_num_list = {}
    with codecs.open(inFilePath, "rb", "utf-8") as input_file:
        for line in input_file.readlines():
            temp = line.strip().split('\t')
            tool_num_list[temp[2]] = tool_num_list.get(temp[2], 1) + 1
        print len(tool_num_list)
    dict = sorted(tool_num_list.iteritems(), key=lambda d: d[1], reverse=True)
    for i in range(0,35):
        #print dict[key][0],dict[key][1]
        postTool_35.append(dict[i][0])
    print len(postTool_35)

#static total hour times
uid_num_list = {}
def stockUid(inFilePath):
    with codecs.open(inFilePath, "rb", "utf-8") as input_file:
        for line in input_file.readlines():
            temp = line.strip().split('\t')
            uid_num_list[temp[0]] = uid_num_list.get(temp[0], 1) + 1
        print len(uid_num_list)


#compute the post tool times vector
def staicPostTime(inFilePath,output_path):
    with open(output_path, 'w') as output_file:
        for uid in uid_num_list.keys():
            static_tool = {}
            toolVec = []
            belong_index = ''
            with codecs.open(inFilePath, "rb", "utf-8") as input_file:
                for line in input_file.readlines():
                    temp = line.strip().split('\t')
                    if (temp[0] == uid) and (temp[2] in postTool_35):
                        static_tool[temp[2]] = static_tool.get(temp[2], 0) + 1
                        if belong_index != '':
                            continue
                        else:
                            belong_index = temp[3]
                #print len(static_hour)

            for tool in postTool_35:
                if tool in static_tool.keys():
                    #print static_hour[str(j)],uid_num_list[uid]
                    ratio_tool = round((static_tool[tool] / float(uid_num_list[uid])),2)
                    #print 'ratio_hour ' + str(ratio_hour)
                    toolVec.append(ratio_tool)
                else:
                    toolVec.append(0.0)
            vecItem = re.sub('\[', '',str(toolVec))
            vecItem = re.sub('\]', '', str(vecItem))
            output_file.write(str(uid) +', ' + vecItem +', '+ belong_index +'\n')

if __name__ == "__main__":
    filePath = "D:\\incomeLevelPrediction\\db_file\\train_data_20\\"
    inPath = filePath + "behaver_demographic_data\\behavioral_time_tool.csv"
    outPath = filePath + "behaver_demographic_data\\behavioral_tool_vec.csv"
    postTools(inPath)
    stockUid(inPath)
    staicPostTime(inPath,outPath)