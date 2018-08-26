package fr.poverty.spark.utils

import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.collection.mutable
import scala.io.Source


class DefineLabelFeaturesTask(val idColumn: String, val labelColumn: String, val featureColumn: String, val dropColumns: Array[String], val sourcePath: String) {

  def run(spark: SparkSession, data: DataFrame): DataFrame = {
    defineLabelFeatures(spark, data)
  }

  def readFeatureNames(): Array[String] = {
    val nullFeatures = Source.fromFile(s"$sourcePath/nullFeaturesNames").getLines.toList.head.split(",")
    val yesFeatures = Source.fromFile(s"$sourcePath/yesNoFeaturesNames").getLines.toList.head.split(",")
    var columns = nullFeatures ++ yesFeatures
    dropColumns.foreach(column => columns = columns.filter(_ != column))
    columns
  }

  def defineLabelFeatures(spark: SparkSession, data: DataFrame): DataFrame = {
    val featureColumnBroadcast = spark.sparkContext.broadcast(featureColumn)
    val labelValues = defineLabelValues(spark, data)
    val getDenseVector = udf((values: mutable.WrappedArray[Double]) => UtilsObject.defineDenseVector(values))
    labelValues.withColumn(featureColumnBroadcast.value, getDenseVector(col("values"))).drop("values")
  }

  def defineLabelValues(spark: SparkSession, data: DataFrame): DataFrame = {
    val featuresBroadcast = spark.sparkContext.broadcast(readFeatureNames())
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
