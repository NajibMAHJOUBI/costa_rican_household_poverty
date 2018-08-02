package fr.poverty.spark.classification.stackingMethod

import fr.poverty.spark.utils.LoadDataSetTask
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

import scala.collection.mutable

class StackingMethodTask(val classificationMethod: String,
                         val pathPrediction: List[String], val formatPrediction: String,
                         val pathTrain: String, val formatTrain: String,
                         val pathSave: String,
                         val validationMethod: String,
                         val ratio: Double,
                         val idColumn: String,
                         val labelColumn: String,
                         val predictionColumn: String) {

  var data: DataFrame = _
  var prediction: DataFrame = _

  def getData: DataFrame = data

  def getPrediction: DataFrame = prediction

  def mergeData(spark: SparkSession): StackingMethodTask = {
    data = loadDataLabel(spark)
    pathPrediction.foreach(path => data = data.join(loadDataPredictionByLabel(spark, path, pathPrediction.indexOf(path)), Seq(idColumn)))
    data = data.drop(idColumn)
    this
  }

  def loadDataPredictionByLabel(spark: SparkSession, path: String, index: Int): DataFrame = {
    new LoadDataSetTask(path, format = formatPrediction)
      .run(spark, "")
      .select(col("id"), col(predictionColumn).alias(s"prediction_${index.toString}"))
  }

  def loadDataLabel(spark: SparkSession): DataFrame = {
    new LoadDataSetTask(sourcePath = pathTrain, format=formatTrain).run(spark, "train")
      .select(col(idColumn), col(labelColumn).alias("label"))
  }

  def createLabelFeatures(spark: SparkSession): DataFrame = {
    mergeData(spark)
    val classificationMethodsBroadcast = spark.sparkContext.broadcast(pathPrediction.toArray)
    val features = (p: Row) => {StackingMethodObject.extractValues(p, classificationMethodsBroadcast.value)}
    val rdd = data.rdd.map(p => (p.getInt(p.fieldIndex("label")), features(p)))
    val labelFeatures = spark.createDataFrame(rdd).toDF("label", "values")
    val defineFeatures = udf((p: mutable.WrappedArray[Double]) => Vectors.dense(p.toArray[Double]))
    labelFeatures.withColumn("features", defineFeatures(col("values"))).select("label", "features")
  }

}
