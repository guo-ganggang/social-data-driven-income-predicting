import MySQLdb

def getDBConnection():
    try:
        con = MySQLdb.connect(host = "10.0.109.33", user = 'guoganggang', passwd ='Lifelab268', db='',port=3306, charset='utf8')
        # con = MySQLdb.connect(host = "123.57.237.131", user = 'root', passwd ='admin', db='wangbaoqiang_divorce',port=3306, charset='utf8')
        return (con,con.cursor())
    except MySQLdb.Error:
        raise