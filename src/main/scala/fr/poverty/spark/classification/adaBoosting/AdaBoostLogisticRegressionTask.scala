package fr.poverty.spark.classification.adaBoosting

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{col, lit, sum, udf}

import scala.math.log

class AdaBoostLogisticRegressionTask(val idColumn: String,
                                     val labelColumn: String,
                                     val featureColumn: String,
                                     val predictionColumn: String,
                                     val weightColumn: String) {

  private var model: LogisticRegression = _
  private var numberOfClass: Long = _

  def run(spark: SparkSession, data: DataFrame): Unit = {
    numberOfClass = getNumberOfClass(data)
  }

  def defineModel(): Unit = {
    model = new LogisticRegression()
      .setLabelCol(labelColumn)
      .setFeaturesCol(featureColumn)
      .setPredictionCol(predictionColumn)
      .setWeightCol(weightColumn)
  }

  def getNumberOfObservation(data: DataFrame): Long = data.count()

  def getNumberOfClass(data: DataFrame): Long = data.select(labelColumn).distinct().count()

  def getInitialWeights(data: DataFrame): Double = 1.0 / getNumberOfObservation(data)

  def addInitialWeightColumn(data: DataFrame): DataFrame = {
    val initialWeight = getInitialWeights(data)
    data.withColumn(weightColumn, lit(initialWeight))
  }

  def sumWeight(data: DataFrame): Double = {
    data.agg(sum(weightColumn).alias("sum")).rdd.map(p => p.getDouble(p.fieldIndex("sum"))).collect()(0)
  }

  def computeWeightError(data: DataFrame): Double = {
    val udfFilterLabelPrediction = udf((target: Int, prediction: Int) => target != prediction)
    val sumWeightError = data.filter(udfFilterLabelPrediction(col(labelColumn), col(predictionColumn)))
     .agg(sum("weight").alias("sum"))
      .rdd.map(p => p.getDouble(p.fieldIndex("sum"))).collect()(0)
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
      .select(col(idColumn), col("updateWeight").alias(weightColumn), col(featureColumn))
  }

  def getModel: LogisticRegression = model

}
