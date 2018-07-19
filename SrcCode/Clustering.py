from numpy import array
from math import sqrt
#from pyspark.mllib.clustering import KMeans, KMeansModel (RDD Based)
from pyspark.ml.clustering import KMeans,KMeansModel

spark=None
sqlContext=None
clusters=None


def setSparkContext(sc):
    global spark
    spark = sc


def setSqlContext(Sqlc):
    global sqlContext
    sqlContext = Sqlc


class clustering:

    def __init__(self,dataset):
        self.dataset = dataset
        self.clusters = None
        self.centers = None

    def error(self,point):
        center = self.clusters.centers[clusters.predict(point)]
        return sqrt(sum([x ** 2 for x in (point - center)]))

    def KMeansclustering(self,k):

        print("Clustering for Dataset")
        self.dataset.cube("label").count().orderBy("label").show()

        kmeans = KMeans().setK(k).setSeed(1)
        model = kmeans.fit(self.dataset)

        # Evaluate clustering by computing Within Set Sum of Squared Errors.
        WSSSE = model.computeCost(self.dataset)
        print("Within Set Sum of Squared Errors = " + str(WSSSE))

        # Shows the result.
        centers = model.clusterCenters()
        # print("Cluster Centers: ")
        # for center in centers:
        #     print(center)

        # Save and load model
        #model.write().overwrite().save("target/org/apache/spark/PythonKMeansExample/KMeansModel"+k)
        #sameModel = KMeansModel.load("target/org/apache/spark/PythonKMeansExample/KMeansModel")
        return WSSSE
