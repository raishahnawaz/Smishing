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
    tokenized=tokenized.withColumn("tokens",countTokens(F.col("words")))
    tokenized.select("words","tokens").show()

    print ("After dropping Null Messages: ", tokenized.count())

    #Transforming words into feature vectors and Label as LabelIndex

    from pyspark.ml.feature import HashingTF, IDF, Tokenizer

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
    nb=NaiveBayes()
    nb.setLabelCol("LabelIndex")
    nb.setPredictionCol("Label_Prediction")


    training, test = indexed.randomSplit([0.9, 0.1], seed=11)
    nvModel = nb.fit(training)
    prediction = nvModel.transform(test)


    selected = prediction.select("body", "LabelIndex", "label", "Label_Prediction")
    for row in selected.collect():
        print(row)

    from pyspark.mllib.evaluation import MulticlassMetrics

    predictionAndLabels =  prediction.select("Label_Prediction","LabelIndex").rdd.map(lambda r:(float(r[0]),float(r[1])))

    #predictionAndLabels = test.rdd.map(lambda lp: (float(nvModel.predict(lp.features)), lp.label))
    metrics = MulticlassMetrics(predictionAndLabels)

    precision = metrics.precision()
    recall = metrics.recall()
    f1Score = metrics.fMeasure()
    print("Summary Stats")
    print("Precision = %s" % precision)
    print("Recall = %s" % recall)
    print("F1 Score = %s" % f1Score)

    #Statistics by class
    labels = prediction.rdd.map(lambda lp: lp.label).distinct().collect()
    labelIndices = prediction.rdd.map(lambda lp: lp.LabelIndex).distinct().collect()
    labelIndicesPairs = prediction.rdd.map(lambda lp: (lp.label,lp.LabelIndex)).distinct().collect()

    print(labels)
    print(labelIndices)
    print(labelIndicesPairs)



    for label,labelIndex in sorted(labelIndicesPairs):
        print("\n")
        print("Class %s precision = %s" % (label, metrics.precision(labelIndex)))
        print("Class %s recall = %s" % (label, metrics.recall(labelIndex)))
        print("Class %s F1 Measure = %s" % (label, metrics.fMeasure(labelIndex, beta=1.0)))

    # Weighted stats
    print("Weighted recall = %s" % metrics.weightedRecall)
    print("Weighted precision = %s" % metrics.weightedPrecision)
    print("Weighted F(1) Score = %s" % metrics.weightedFMeasure())
    print("Weighted F(0.5) Score = %s" % metrics.weightedFMeasure(beta=0.5))
    print("Weighted false positive rate = %s" % metrics.weightedFalsePositiveRate)


    spark.stop()
