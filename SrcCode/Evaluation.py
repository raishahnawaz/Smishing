from pyspark.ml.classification import LogisticRegression,NaiveBayes

def NaiveBayesEvaluation(TransformedDataset):

    nb=NaiveBayes()
    nb.setLabelCol("LabelIndex")
    nb.setPredictionCol("Label_Prediction")
    training, test = TransformedDataset.randomSplit([0.9, 0.1], seed=11)
    nvModel = nb.fit(training)
    prediction = nvModel.transform(test)

    selected = prediction.select("body", "LabelIndex", "label", "Label_Prediction")
    for row in selected.collect():
        print(row)

    from pyspark.mllib.evaluation import MulticlassMetrics

    predictionAndLabels = prediction.select("Label_Prediction", "LabelIndex").rdd.map(
        lambda r: (float(r[0]), float(r[1])))

    # predictionAndLabels = test.rdd.map(lambda lp: (float(nvModel.predict(lp.features)), lp.label))
    metrics = MulticlassMetrics(predictionAndLabels)

    precision = metrics.precision()
    recall = metrics.recall()
    f1Score = metrics.fMeasure()
    print("Summary Stats")
    print("Precision = %s" % precision)
    print("Recall = %s" % recall)
    print("F1 Score = %s" % f1Score)

    # Statistics by class
    labels = prediction.rdd.map(lambda lp: lp.label).distinct().collect()
    labelIndices = prediction.rdd.map(lambda lp: lp.LabelIndex).distinct().collect()
    labelIndicesPairs = prediction.rdd.map(lambda lp: (lp.label, lp.LabelIndex)).distinct().collect()

    print(labels)
    print(labelIndices)
    print(labelIndicesPairs)

    for label, labelIndex in sorted(labelIndicesPairs):
        print("\n Class %s precision = %s" % (label, metrics.precision(labelIndex)))
        print("Class %s recall = %s" % (label, metrics.recall(labelIndex)))
        print("Class %s F1 Measure = %s" % (label, metrics.fMeasure(labelIndex, beta=1.0)),"\n")

    # Weighted stats
    print("Weighted recall = %s" % metrics.weightedRecall)
    print("Weighted precision = %s" % metrics.weightedPrecision)
    print("Weighted F(1) Score = %s" % metrics.weightedFMeasure())
    print("Weighted F(0.5) Score = %s" % metrics.weightedFMeasure(beta=0.5))
    print("Weighted false positive rate = %s" % metrics.weightedFalsePositiveRate)