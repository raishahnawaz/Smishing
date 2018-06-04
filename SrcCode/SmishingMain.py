from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark import SparkContext, SparkConf,SQLContext
from pyspark.sql import functions as F
import sys
from operator import add
from pyspark.sql import SparkSession
import re
from pydoc import  help
from io import StringIO

from operator import is_not
from functools import partial
import importlib

importlib.reload(sys)
sys.getdefaultencoding()
from os import environ
import numpy
from collections import namedtuple


def loadTextFile_1(fileAddress):
    rawTextRdd = spark.sparkContext.textFile(fileAddress)
    user= namedtuple("user",
                          ['user_id', 'device_id','phone_number', 'phone_number_user_entered', 'phone_model', 'phone_os', 'device_api', 'app_version', 'date_created', 'created_by',
                           'date_modified','modified_by','last_sync_date','last_ping_date','label_counts','user_email'])

    UserData = rawTextRdd.map(lambda para: para.split("|")).filter(lambda line: line[0] != 'user_id') \
        .map(lambda line: user(line[0], line[1], line[2],line[3], line[4],line[5],line[6], line[7],line[8],line[9],line[10],line[11],line[12],line[13]
                                   ,line[14],line[15]))

    return UserData.toDF()

def loadTextFiles_2(fileAddress):
    df = spark.read.load(fileAddress,format="csv", sep="|", inferSchema="true", header="true")
    return df


if __name__ == '__main__':

    print ("Hello");

    print(environ['SPARK_HOME'])
    print(environ['JAVA_HOME'])
    try:
        print(environ['PYSPARK_SUBMIT_ARGS'])
    except:
        print(
            "no problem with PYSPARK_SUBMIT_ARGS")  # https://github.com/ContinuumIO/anaconda-issues/issues/1276#issuecomment-277355043

    conf = SparkConf().setAppName("Smishing")
    spark = SparkSession \
        .builder \
        .appName("Smishing") \
        .config(conf=conf) \
        .getOrCreate()


    sqlContext=SQLContext(sparkContext=spark)
    Users=loadTextFile_1("C:/Users/Rai Shahnawaz/Desktop/FINTECH PROJECTS/SMS Fraud Detection in DFS Pakistan/Data And Statistics/Data/Data 06-04-2018/users.txt")
    Threads=loadTextFiles_2("C:/Users/Rai Shahnawaz/Desktop/FINTECH PROJECTS/SMS Fraud Detection in DFS Pakistan/Data And Statistics/Data/Data 06-04-2018/threads.txt")
    Messages=loadTextFiles_2("C:/Users/Rai Shahnawaz/Desktop/FINTECH PROJECTS/SMS Fraud Detection in DFS Pakistan/Data And Statistics/Data/Data 06-04-2018/messages.txt")


    Users.show()
    Threads.show()
    Messages.show()


    #Creating Temporary Views for these dataframes : Views Life will end with the sparksession termination

    Users.createOrReplaceTempView("users")
    Threads.createOrReplaceTempView("threads")
    Messages.createOrReplaceTempView("messages")

    UsersCount="select count(*), date(date_created) from users group by 2 order by 1 desc"
    DateWiseUsers=spark.sql(UsersCount)
    DateWiseUsers.show()


