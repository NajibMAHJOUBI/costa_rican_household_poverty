package fr.poverty.spark.clustering.semiSupervised


import org.apache.spark.ml.Model
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.collection.Map


class SemiSupervisedTask(val idColumn: String,
                         val labelColumn: String,
                         val featuresColumn: String,
                         val pathSave: String) {

  val predictionColumn: String = "prediction"
  var model: Model[_] = _
  var prediction: DataFrame = _
  var submission: DataFrame = _
  var mapPredictionLabel: Map[Int, Int] = _

  def getNumberOfClusters(data: DataFrame): Int = {
    data.select(labelColumn).distinct().count().toInt
  }

  def computePrediction(data: DataFrame): SemiSupervisedTask = {
    prediction = model.transform(data)
    this
  }

  def computeSubmission(spark: SparkSession, data: DataFrame): SemiSupervisedTask = {
    val mapPredictionLabelBroadcast = spark.sparkContext.broadcast(mapPredictionLabel)
    val getTarget = udf((prediction: Int) => mapPredictionLabelBroadcast.value(prediction))
    submission = model.transform(data)
      .withColumn(labelColumn, getTarget(col(predictionColumn)))
        .select(idColumn, labelColumn)
    this
  }

  def defineMapPredictionLabel(spark: SparkSession, data: DataFrame): SemiSupervisedTask = {
    val predictionColumnBroadcast = spark.sparkContext.broadcast(predictionColumn)
    val labelColumnBroadcast = spark.sparkContext.broadcast(labelColumn)
    mapPredictionLabel = prediction
      .rdd
      .map(row => (row.getInt(row.fieldIndex(predictionColumnBroadcast.value)),
        List(row.getInt(row.fieldIndex(labelColumnBroadcast.value)))))
      .reduceByKey(_ ++ _)
      .map(p => (p._1, p._2.groupBy(identity).mapValues(_.size).maxBy(_._2)._1))
      .collectAsMap()
    this
  }

  def savePrediction(): SemiSupervisedTask = {
    prediction
      .coalesce(1)
      .write.mode("overwrite")
      .parquet(s"$pathSave/prediction")
    this
  }

  def saveSubmission(): SemiSupervisedTask = {
    submission
      .select(col(idColumn), col(labelColumn).cast(IntegerType))
      .repartition(1)
      .write
      .option("header", "true")
      .option("delimiter", ",")
      .mode("overwrite")
      .csv(s"$pathSave/submission")
    this
  }

  def getPrediction: DataFrame = prediction
}
