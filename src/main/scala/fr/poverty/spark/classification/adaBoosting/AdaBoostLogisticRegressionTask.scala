package fr.poverty.spark.classification.adaBoosting

import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.sql.functions.{col, lit, sum, udf}
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.math.log

class AdaBoostLogisticRegressionTask(val idColumn: String,
                                     val labelColumn: String,
                                     val featureColumn: String,
                                     val predictionColumn: String,
                                     val weightColumn: String,
                                     val numberOfWeakClassifier: Int) {

  private var model: LogisticRegression = _
  private var numberOfClass: Long = _
  private var numberOfObservation: Long = _
  private var initialObservationWeight: Double = _
  private var weightWeakClassifierList: List[Double] = List()
  private var weakClassifierList: List[LogisticRegressionModel] = List()

  def run(spark: SparkSession, data: DataFrame): AdaBoostLogisticRegressionTask = {
    computeNumberOfObservation(data)
    computeNumberOfClass(data)
    computeInitialObservationWeight(data)
    defineModel()
    loopWeakClassifier(spark, data)
    computePrediction(spark, data)
    this
  }

  def loopWeakClassifier(spark: SparkSession, data: DataFrame): AdaBoostLogisticRegressionTask = {
    var weightError: Double = 1.0
    var index: Int = 1
    var weightData: DataFrame = addInitialWeightColumn(data)
    while(weightError > 1e-6 && index <= numberOfWeakClassifier) {
      val modelFitted = model.fit(weightData)
      weakClassifierList = weakClassifierList ++ List(modelFitted)
      weightData = modelFitted.transform(weightData)
      weightError = computeWeightError(weightData)
      val weightWeakClassifier = computeWeightWeakClassifier(weightData, weightError)
      weightWeakClassifierList = weightWeakClassifierList ++ List(weightWeakClassifier)
      weightData = updateWeightObservation(spark, weightData, weightWeakClassifier)
      index += 1
    }
    this
  }

  def computePrediction(spark: SparkSession, data: DataFrame): DataFrame = {
    val numberOfClassifier = weightWeakClassifierList.length
    val dataWeight = addUnitaryWeightColumn(data)
    var idDataSet: DataFrame = data.select(idColumn)
    val numberOfClassifierBroadcast = spark.sparkContext.broadcast(numberOfClassifier)
    val idColumnBroadcast = spark.sparkContext.broadcast(idColumn)

    (0 until numberOfClassifier).foreach(index => {
      println(s"weight classifier: ${weightWeakClassifierList(index)}")
      val weightBroadcast = spark.sparkContext.broadcast(weightWeakClassifierList(index))
      val udfMerge = udf((prediction: Double) => AdaBoostingObject.mergePredictionWeight(prediction, weightBroadcast.value))
      val weakTransform = weakClassifierList(index).transform(dataWeight).select(idColumn, predictionColumn)
        .withColumn(s"prediction_$index", udfMerge(col(predictionColumn))).drop(predictionColumn)
        .select(col(idColumn), col(s"prediction_$index"))
      idDataSet = idDataSet.join(weakTransform, Seq(idColumn))
    })

    val rdd = idDataSet.rdd.map(p => (p.getString(p.fieldIndex(idColumnBroadcast.value)), AdaBoostingObject.mergePredictionWeightList(p, numberOfClassifierBroadcast.value)))
    spark.createDataFrame(rdd).toDF(idColumn, predictionColumn)
  }

  def defineModel(): AdaBoostLogisticRegressionTask = {
    model = new LogisticRegression()
      .setLabelCol(labelColumn)
      .setFeaturesCol(featureColumn)
      .setPredictionCol(predictionColumn)
      .setWeightCol(weightColumn)
    this
  }

  def computeNumberOfObservation(data: DataFrame): AdaBoostLogisticRegressionTask = {
    numberOfObservation = data.count()
    this
  }

  def computeNumberOfClass(data: DataFrame): AdaBoostLogisticRegressionTask = {
    numberOfClass = data.select(labelColumn).distinct().count()
    this
  }

  def computeInitialObservationWeight(data: DataFrame): AdaBoostLogisticRegressionTask = {
    initialObservationWeight = 1.0 / numberOfObservation
    this
  }

  def addInitialWeightColumn(data: DataFrame): DataFrame = {
    data.withColumn(weightColumn, lit(initialObservationWeight))
  }

  def addUnitaryWeightColumn(data: DataFrame): DataFrame = {
    data.withColumn(weightColumn, lit(1.0))
  }

  def sumWeight(data: DataFrame): Double = {
    val resultSum = if(data.count() == 0){
      0.0
    } else {
      data
        .agg(sum(weightColumn).alias("sum"))
        .rdd
        .map(p => p.getDouble(p.fieldIndex("sum")))
        .collect()(0)
    }
    resultSum
  }

  def computeWeightError(data: DataFrame): Double = {
    val udfFilterLabelPrediction = udf((label: Double, prediction: Double) => label != prediction)
    val sumWeightError = sumWeight(data.filter(udfFilterLabelPrediction(col(labelColumn), col(predictionColumn))))
    val weightSum = sumWeight(data)
    sumWeightError / weightSum
  }

  def computeWeightWeakClassifier(data: DataFrame, error: Double): Double = {
    log((1 - error) / error) + log(numberOfClass - 1)
  }

  def updateWeightObservation(spark: SparkSession, data: DataFrame, weightClassifier: Double): DataFrame = {
    val weightClassifierBroadcast = spark.sparkContext.broadcast(weightClassifier)
    val udfExponentialWeight = udf((target: Int, prediction: Int) => AdaBoostingObject.exponentialWeightObservation(target, prediction, weightClassifierBroadcast.value))
    data.withColumn("updateWeight", udfExponentialWeight(col(labelColumn), col(predictionColumn)))
      .select(col(idColumn), col(labelColumn), col("updateWeight").alias(weightColumn), col(featureColumn))
  }

  def getModel: LogisticRegression = model

  def getNumberOfClass: Long = numberOfClass

  def getNumberOfObservation: Long = numberOfObservation

  def getInitialObservationWeight: Double = initialObservationWeight

  def getWeakClassifierList: List[Double] = weightWeakClassifierList
}
