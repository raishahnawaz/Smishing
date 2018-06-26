from collections import namedtuple

spark=None

def setSparkContext(sc):
    global spark
    spark=sc

def SubsetSelection(dataset,n):
    count = dataset.count()
    if (count > n):
        numberOfRecords=n
    else:
        numberOfRecords=count
    return dataset.sample(False, 1.0 * numberOfRecords / count, seed=1234).limit(n)

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

def loadExcel_file(fileAddress):
    df = spark.read.format("com.crealytics.spark.excel").\
    option("location", fileAddress). \
        option("spark.read.simpleMode", "true"). \
        option("treatEmptyValuesAsNulls", "true"). \
        option("addColorColumns", "false"). \
        option("useHeader", "true"). \
        option("inferschema", "true"). \
        load("com.databricks.spark.csv")
    return df