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
from pyspark.sql.types import *
from operator import is_not
from functools import partial
import importlib

importlib.reload(sys)
sys.getdefaultencoding()
from os import environ
import numpy
from collections import namedtuple
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.sql.functions import col,udf,when
from pyspark.sql.types import IntegerType, StringType,ArrayType

def loadTextFile_1(fileAddress):
    rawTextRdd = spark.sparkContext.textFile(fileAddress)
    user= namedtuple("user",
                          ['user_id', 'device_id','phone_number', 'phone_number_user_entered', 'phone_model', 'phone_os', 'device_api', 'app_version', 'date_created', 'created_by',
                           'date_modified','modified_by','last_sync_date','last_ping_date','label_counts','user_email'])

    UserData = rawTextRdd.map(lambda para: para.split("|")).filter(lambda line: line[0] != 'user_id') \
        .map(lambda line: user(line[0], line[1], line[2],line[3], line[4],line[5],line[6], line[7],line[8],line[9],line[10],line[11],line[12],line[13]
                                   ,line[14],line[15]))

    return UserData.toDF()

def loadTextFiles_2(fileAddress,delimiter):
    df = spark.read.load(fileAddress,format="csv", sep=delimiter, inferSchema="true", header="true")
    return df



def Truecounts(wordsList):
    if (wordsList.isEmpty()) | (wordsList is None):
        l=0
    else:
            l=len(wordsList)
    return  l

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
    Threads=loadTextFiles_2("C:/Users/Rai Shahnawaz/Desktop/FINTECH PROJECTS/SMS Fraud Detection in DFS Pakistan/Data And Statistics/Data/Data 06-04-2018/threads.txt","|")
    #Messages=loadTextFiles_2("C:/Users/Rai Shahnawaz/Desktop/FINTECH PROJECTS/SMS Fraud Detection in DFS Pakistan/Data And Statistics/Data/Data 06-04-2018/messages.txt","|")

    query = """
      (select id, thread_uid_id, creator, body from sms_storage_services_smsmms) foo
    """
    Messages=sqlContext.read.format("jdbc").options(url="jdbc:sqlite:C:/Users/Rai Shahnawaz/Desktop/FINTECH PROJECTS/SMS Fraud Detection in DFS Pakistan/Data And Statistics/Data/Data 06-04-2018/db.sqlite3", \
                                           driver="org.sqlite.JDBC",
                                           dbtable=query).load()



    #Users.show()
    #Threads.show()
    Messages.show()

    #Creating Temporary Views for these dataframes : Views Life will end with the sparksession termination

    Users.createOrReplaceTempView("users")
    Threads.createOrReplaceTempView("threads")
    Messages.createOrReplaceTempView("messages")


    UsersCount="select count(*), date(date_created) from users group by 2 order by 1 desc"
    DateWiseUsers=spark.sql(UsersCount)
    DateWiseUsers.show()

    Query2="select m.*, t.label from threads t inner join messages m on t.thread_uid=m.thread_uid_id"
    LabeledMessages=spark.sql(Query2)

    print ("Users Count: ", Users.count())
    print ("Threads Count: ", Threads.count())
    print ("Count of Messages: ", Messages.count())
    print ("Count of Labeled Messages: ", LabeledMessages.count())



    #Tokenzing Sentences into words

    tokenizer = Tokenizer(inputCol="body", outputCol="words")
    countTokens = udf(lambda words: len(words))
    tokenized = tokenizer.transform(LabeledMessages.na.drop(subset=["body"]))
    #print("Tokenized Words ",tokenized.select("body",col("words")).show())
    #tokenized=tokenized.withColumn("TestColumn",countTokens(F.lit("words")))
    tokenized=tokenized.withColumn("tokens",countTokens(F.col("words")))
    tokenized.select("words","tokens").show()

    print ("After dropping Null Labeled Messages: ", LabeledMessages.count())

    #Transforming words into feature vectors

    from pyspark.ml.feature import HashingTF, IDF, Tokenizer

    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=2048)
    featurizedData = hashingTF.transform(tokenized)
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)
    rescaledData.select("label", "features").show()


