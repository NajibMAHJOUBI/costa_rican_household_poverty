package fr.poverty.spark.stat

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.feature.ChiSqSelector

import scala.collection.mutable

class ChiSquareTask(val labelColumn: String, val featuresColumn: List[String], alpha: Double) {

  def run(spark: SparkSession, data: DataFrame): DataFrame = {
    val labelFeatures = defineLabelFeatures(spark, data)
    ChiSquareSelector(labelFeatures)
  }

  def defineLabelValues(spark: SparkSession, data: DataFrame): RDD[(Double, Array[Double])] = {
    val labelBroadcast = spark.sparkContext.broadcast(labelColumn)
    val featuresBroadcast = spark.sparkContext.broadcast(featuresColumn)
    data.rdd.map(row => (row.getInt(row.fieldIndex(labelBroadcast.value)).toDouble, StatObject.extractValues(row, featuresBroadcast.value)))
  }

  def defineLabelFeatures(spark: SparkSession, data: DataFrame): DataFrame = {
    val defineFeatures = udf((p: mutable.WrappedArray[Double]) => Vectors.dense(p.toArray[Double]))
    val rdd = defineLabelValues(spark, data)
    spark.createDataFrame(rdd).toDF("label", "values")
      .withColumn("features", defineFeatures(col("values")))
      .select("label", "features")
  }

  def ChiSquareSelector(data: DataFrame): DataFrame = {
    val chiSelector = new ChiSqSelector().setSelectorType("fpr").setFpr(alpha).setLabelCol("label").setFeaturesCol("features").setOutputCol("selectedFeatures")
    chiSelector.fit(data).transform(data)
  }

}
