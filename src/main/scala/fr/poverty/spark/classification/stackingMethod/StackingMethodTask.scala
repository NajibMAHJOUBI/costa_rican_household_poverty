package fr.poverty.spark.classification.stackingMethod

import fr.poverty.spark.utils.{LoadDataSetTask, StringIndexerTask}
import org.apache.spark.ml.feature.StringIndexerModel
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

import scala.collection.mutable

class StackingMethodTask(val idColumn: String, val labelColumn: String, val predictionColumn: String,
                         val pathPrediction: List[String], val formatPrediction: String,
                         val pathTrain: String, val formatTrain: String,
                         val pathStringIndexer: String, val pathSave: String,
                         val validationMethod: String, val ratio: Double) {

  var data: DataFrame = _
  var prediction: DataFrame = _
  var labelFeatures: DataFrame = _

  def getData: DataFrame = data

  def getPrediction: DataFrame = prediction

  def getLabelFeatures: DataFrame = labelFeatures

  def mergeData(spark: SparkSession): StackingMethodTask = {
    data = loadDataLabel(spark)
    pathPrediction.foreach(path => data = data.join(loadDataPredictionByLabel(spark, path, pathPrediction.indexOf(path)), Seq(idColumn)))
    data = data//.drop(idColumn)
    this
  }

  def loadDataPredictionByLabel(spark: SparkSession, path: String, index: Int): DataFrame = {
    new LoadDataSetTask(path, format = formatPrediction)
      .run(spark, "")
      .select(col(idColumn), col(predictionColumn).alias(s"prediction_${index.toString}"))
  }

  def loadDataLabel(spark: SparkSession): DataFrame = {
    val data = new LoadDataSetTask(sourcePath = pathTrain, format=formatTrain).run(spark, "train")
      .select(col(idColumn), col(labelColumn))
    val stringIndexerModel: StringIndexerModel = loadStringIndexerModel()
    stringIndexerModel.transform(data)
  }

  def createLabelFeatures(spark: SparkSession): DataFrame = {
    mergeData(spark)
    val idColumnBroadcast = spark.sparkContext.broadcast(idColumn)
    val classificationMethodsBroadcast = spark.sparkContext.broadcast(pathPrediction.toArray)
    val features = (p: Row) => {StackingMethodObject.extractValues(p, classificationMethodsBroadcast.value)}
    val rdd = data.rdd.map(p => (p.getString(p.fieldIndex(idColumnBroadcast.value)), p.getDouble(p.fieldIndex("label")), features(p)))
    val labelFeatures = spark.createDataFrame(rdd).toDF(idColumn, labelColumn, "values")
    val defineFeatures = udf((p: mutable.WrappedArray[Double]) => Vectors.dense(p.toArray[Double]))
    labelFeatures.withColumn("features", defineFeatures(col("values"))).select(idColumn, labelColumn, "features")
  }

  def loadStringIndexerModel():  StringIndexerModel = {
    new StringIndexerTask("", "", "").loadModel(pathStringIndexer)
  }

}
