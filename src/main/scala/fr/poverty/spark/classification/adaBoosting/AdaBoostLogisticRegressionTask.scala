package fr.poverty.spark.classification.adaBoosting

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{col, lit, sum, udf}

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
  private var weakClassifierList: List[Double] = List()

  def run(spark: SparkSession, data: DataFrame): AdaBoostLogisticRegressionTask = {
    computeNumberOfObservation(data)
    computeNumberOfClass(data)
    computeInitialObservationWeight(data)
    defineModel()
    loopWeakClassifier(spark, data)
    this
  }

  def loopWeakClassifier(spark: SparkSession, data: DataFrame): AdaBoostLogisticRegressionTask = {
    var weightData: DataFrame = addInitialWeightColumn(data)
    (1 to numberOfWeakClassifier).foreach( nb => {
      println(s"classifier: ${nb}")
      weightData.show()
      val modelFitted = model.fit(weightData)
      weightData = modelFitted.transform(weightData)
      weightData.show()
      println(s"error: ${computeWeightError(weightData)}")
      val weightWeakClassifier = computeWeightWeakClassifier(weightData)
      weakClassifierList = weakClassifierList ++ List(weightWeakClassifier)
      weightData = updateWeightObservation(spark, weightData, weightWeakClassifier)
    })
    this
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

  def computeWeightWeakClassifier(data: DataFrame): Double = {
    val weightError = computeWeightError(data)
    log((1 - weightError) / weightError) + log(numberOfClass - 1)
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

  def getWeakClassifierList: List[Double] = weakClassifierList
}
