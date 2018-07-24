package fr.poverty.spark.utils

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{col, udf}

import scala.io.Source

class DefineLabelFeaturesTask(val labelColumn: String) {

  private var featureNames: Array[String] = _

  def run(data: DataFrame): Unit = {
    featureNames = readFeatureNames()
  }

  def setFeatureNames(features: Array[String]): Unit = {
    featureNames = features
  }

  def readFeatureNames(): Array[String] = {
    Source.fromFile("src/main/resources/featuresNames").getLines.toList(0).split(",")
  }

  def defineLabelFeatures(spark: SparkSession, data: DataFrame): DataFrame = {
    val labelValues = defineLabelValues(spark, data)
    val getDenseVector = udf((values: List[Double]) => UtilsObject.defineDenseVector(values))
    labelValues.withColumn("features", getDenseVector(col("values"))).drop("values")
  }

  def defineLabelValues(spark: SparkSession, data: DataFrame): DataFrame = {
    val featuresBroadcast = spark.sparkContext.broadcast(featureNames)
    val labelBroadcast = spark.sparkContext.broadcast(labelColumn)
    val rdd = data.rdd
      .map(p => (p.getInt(p.fieldIndex(labelBroadcast.value)),
        featuresBroadcast.value.map(feature => p.getDouble(p.fieldIndex(feature))).toList))
    spark.createDataFrame(rdd).toDF(labelColumn, "values")
  }

}
