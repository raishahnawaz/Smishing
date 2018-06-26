from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression,NaiveBayes
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
import Evaluation
from LoadData import Dataload
import ShareSparkVariables as SSV
from pyspark.ml.feature import HashingTF, IDF, Tokenizer



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

    SSV.ShareSparkContext(spark)

    #spark.sparkContext.setLogLevel("OFF")

    sqlContext=SQLContext(sparkContext=spark)
    Users=Dataload.loadTextFile_1("C:/Users/Rai Shahnawaz/Desktop/FINTECH PROJECTS/SMS Fraud Detection in DFS Pakistan/Data And Statistics/Data/Data 06-04-2018/users.txt")
    Threads=Dataload.loadTextFiles_2("C:/Users/Rai Shahnawaz/Desktop/FINTECH PROJECTS/SMS Fraud Detection in DFS Pakistan/Data And Statistics/Data/Data 06-04-2018/threads.txt","|")
    #Messages=loadTextFiles_2("C:/Users/Rai Shahnawaz/Desktop/FINTECH PROJECTS/SMS Fraud Detection in DFS Pakistan/Data And Statistics/Data/Data 06-04-2018/messages.txt","|")

    query = """
      (select id, thread_uid_id, creator, body from sms_storage_services_smsmms) foo
    """
    Messages=sqlContext.read.format("jdbc").options(url="jdbc:sqlite:C:/Users/Rai Shahnawaz/Desktop/FINTECH PROJECTS/SMS Fraud Detection in DFS Pakistan/Data And Statistics/Data/Data 06-04-2018/db.sqlite3", \
                                           driver="org.sqlite.JDBC",
                                           dbtable=query).load()

    TrueFrauds=Dataload.loadExcel_file("C:/Users/Rai Shahnawaz/Desktop/FINTECH PROJECTS/SMS Fraud Detection in DFS Pakistan/Data And Statistics/Data/true_fraud.xlsx")

    #TrueFrauds=TrueFrauds.select("id", "thread_uid_id", "creator", "body","label")
    #TrueFraudsSelected=TrueFrauds.select("id","body")
    #TrueFraudsSelected.write.option("sep", "|").option("header", "true").csv("C:/Users/Rai Shahnawaz/Desktop/FINTECH PROJECTS/SMS Fraud Detection in DFS Pakistan/Data And Statistics/Data/TrueFrauds")
    #TrueFraudsSelected.toPandas().to_csv("C:/Users/Rai Shahnawaz/Desktop/FINTECH PROJECTS/SMS Fraud Detection in DFS Pakistan/Data And Statistics/Data/LocalTrueFrauds", sep='|')
    #Users.show()
    #Threads.show()
    #Messages.show()

    #Creating Temporary Views for these dataframes : Views Life will end with the sparksession termination

    Users.createOrReplaceTempView("users")
    Threads.createOrReplaceTempView("threads")
    Messages.createOrReplaceTempView("messages")

    # DateWise Users Count
    UsersCount="select count(*), date(date_created) from users group by 2 order by 1 desc"
    DateWiseUsers=spark.sql(UsersCount)
    DateWiseUsers.show()

    # Loading Labeled Messages
    Query2="select m.*, t.label from threads t inner join messages m on t.thread_uid=m.thread_uid_id"
    Query3="select m.*, t.label from threads t inner join messages m on t.thread_uid=m.thread_uid_id where t.label is not NULL"

    OverallMessages=spark.sql(Query2)
    LabeledMessages=spark.sql(Query3)

    TrueLabeledMessages=LabeledMessages.filter(col('label').isin(['SPAM','OK']))
    TrueLabeledMessages=TrueLabeledMessages.union(TrueFrauds.select("id", "thread_uid_id", "creator", "body","label"))

    print ("Users Count: ", Users.count())
    print ("Threads Count: ", Threads.count())
    print ("Count of Messages: ", Messages.count())
    print ("Count of Labeled Messages: ", LabeledMessages.count())
    print ("Count of True Labeled Messages: ", TrueLabeledMessages.count())
    print ("Total True Labeled Fraudulent Messages:", TrueLabeledMessages.filter(col('label').isin(['FRAUD'])).count())

    #Tokenzing Sentences into words

    tokenizer = Tokenizer(inputCol="body", outputCol="words")
    countTokens = udf(lambda words: len(words))
    tokenized = tokenizer.transform(LabeledMessages.na.drop(subset=["body"]))
    tokenized=tokenized.withColumn("tokens",countTokens(F.col("words")))
    tokenized.select("words","tokens").show()

    print ("After dropping Null Messages: ", tokenized.count())

    #Transforming words into feature vectors and Label as LabelIndex

    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")
    featurizedData = hashingTF.transform(tokenized)
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)
    rescaledData.cube("label").count().orderBy("label").show()

    from pyspark.ml.feature import StringIndexer
    indexer = StringIndexer(inputCol="label", outputCol="LabelIndex")
    indexed = indexer.fit(rescaledData).transform(rescaledData.na.drop(subset=["label"]))

    print ("Labeled Messages: ", indexed.count())
    # print ("OK Messages: ", indexed["label"]=="OK")
    # print ("FRAUD Messages: ", indexed.count(indexed["label"]=="FRAUD"))
    # print ("SPAM Messages: ", indexed.count(indexed["label"]=="SPAM"))

    indexed.cube("label").count().orderBy("label").na.drop(subset=["label"]).show()

    print ("Indexed Schema: ", indexed.schema)
    print ("Labeled Data: ", indexed.count())


    indexed.select("LabelIndex", "features").show()

    TruePipeline=Pipeline()
    TruePipeline.setStages([tokenizer,hashingTF,idf,indexer])
    TrueFraudsModel=TruePipeline.fit(TrueLabeledMessages.na.drop(subset=["body"]))

    TrueFraudsIndexed=TrueFraudsModel.transform(TrueLabeledMessages.na.drop(subset=["body"]))
    print ("TrueFrauds Indexed Schema: ", TrueFraudsIndexed.schema)
    print ("TrueFrauds Count: ", TrueFraudsIndexed.count())
    TrueFraudsIndexed.select("LabelIndex", "features").show()

    #Evaluation.NaiveBayesEvaluation(indexed)
    print ("Evaluation With true labeled fraud")
    Evaluation.NaiveBayesEvaluation(TrueFraudsIndexed)

    #2  Classification with Subset


    OK=indexed.filter(col('label').isin(['OK']))
    SPAM=indexed.filter(col('label').isin(['SPAM']))
    FRAUD=indexed.filter(col('label').isin(['FRAUD']))
    print ("\n OK Count: ", OK.count())
    OKMessages=Dataload.SubsetSelection(OK,FRAUD.count())
    SPAMMessages=Dataload.SubsetSelection(SPAM,FRAUD.count())
    print ("\n OK Messages Count: ", OKMessages.count())
    BalancedDataFrame = OKMessages.union(FRAUD).union(SPAMMessages)
    print ("Subset Data Count: ", BalancedDataFrame.count())
    #df.filter(col('label').isin(['FRAUD'])).show()
    print ("Evaluation with Balanced Data Frame")
    Evaluation.NaiveBayesEvaluation(BalancedDataFrame)

    # 2  Classification with True Label Subset

    TOK = TrueFraudsIndexed.filter(col('label').isin(['OK']))
    TSPAM = TrueFraudsIndexed.filter(col('label').isin(['SPAM']))
    TFRAUD = TrueFraudsIndexed.filter(col('label').isin(['FRAUD']))
    print("\n OK Count: ", OK.count())
    TOKMessages = Dataload.SubsetSelection(TOK, TFRAUD.count())
    TSPAMMessages = Dataload.SubsetSelection(TSPAM, TFRAUD.count())
    print("\n OK Messages Count: ", TOKMessages.count())
    TBalancedDataFrame = TOKMessages.union(TFRAUD).union(TSPAMMessages)
    print("Subset Data Count: ", TBalancedDataFrame.count())
    # df.filter(col('label').isin(['FRAUD'])).show()
    print ("Evaluation with True Label Balanced Data Frame")
    Evaluation.NaiveBayesEvaluation(TBalancedDataFrame)

    spark.stop()
