package fr.poverty.spark.utils

import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.collection.mutable.WrappedArray
import scala.io.Source


class DefineLabelFeaturesTask(val idColumn: String, val labelColumn: String, val sourcePath: String) {

  private var featureNames: Array[String] = _

  def run(spark: SparkSession, data: DataFrame): DataFrame = {
    featureNames = readFeatureNames(sourcePath)
    defineLabelFeatures(spark, data)
  }

  def setFeatureNames(features: Array[String]): Unit = {
    featureNames = features
  }

  def getSourcePath: String = {
    sourcePath
  }

  def readFeatureNames(path: String): Array[String] = {
    val nullFeatures = Source.fromFile(s"$path/nullFeaturesNames").getLines.toList(0).split(",")
    val yesFeatures = Source.fromFile(s"$path/yesNoFeaturesNames").getLines.toList(0).split(",")
    nullFeatures ++ yesFeatures
  }

  def defineLabelFeatures(spark: SparkSession, data: DataFrame): DataFrame = {
    val labelValues = defineLabelValues(spark, data)
    val getDenseVector = udf((values: WrappedArray[Double]) => UtilsObject.defineDenseVector(values))
    labelValues.withColumn("features", getDenseVector(col("values"))).drop("values")
  }

  def defineLabelValues(spark: SparkSession, data: DataFrame): DataFrame = {
    val featuresBroadcast = spark.sparkContext.broadcast(featureNames)
    val idBroadcast = spark.sparkContext.broadcast(idColumn)
    if (labelColumn == "") {
      val rdd = data.rdd.map(p => (p.getString(p.fieldIndex(idBroadcast.value)), featuresBroadcast.value.map(feature => p.getDouble(p.fieldIndex(feature))).toList))
      spark.createDataFrame(rdd).toDF(idColumn, "values")
    } else {
      val labelBroadcast = spark.sparkContext.broadcast(labelColumn)
      val rdd = data.rdd.map(p => (p.getString(p.fieldIndex(idBroadcast.value)), p.getInt(p.fieldIndex(labelBroadcast.value)), featuresBroadcast.value.map(feature => p.getDouble(p.fieldIndex(feature))).toList))
      spark.createDataFrame(rdd).toDF(idColumn, labelColumn, "values")
    }
  }

}
