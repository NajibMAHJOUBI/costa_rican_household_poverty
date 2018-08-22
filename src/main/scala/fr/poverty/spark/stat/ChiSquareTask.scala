package fr.poverty.spark.stat

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.feature.{ChiSqSelector, ChiSqSelectorModel}

import scala.collection.mutable

class ChiSquareTask(val idColumn: String, val labelColumn: String, val featuresColumn: List[String], val featureColumn: String, alpha: Double) {

  private var model: ChiSqSelectorModel = _
  private var trainLabelFeatures: DataFrame = _
  private var testLabelFeatures: DataFrame = _

  def run(spark: SparkSession, train: DataFrame, test: DataFrame): ChiSquareTask = {
    defineLabelFeatures(spark, train, "train")
    defineLabelFeatures(spark, test, "test")
    fit(trainLabelFeatures)
    this
  }

  def defineTrainLabelValues(spark: SparkSession, data: DataFrame): RDD[(String, Double, Array[Double])] = {
    val idBroadcast = spark.sparkContext.broadcast(idColumn)
    val labelBroadcast = spark.sparkContext.broadcast(labelColumn)
    val featuresBroadcast = spark.sparkContext.broadcast(featuresColumn)
    data.rdd.map(row => (row.getString(row.fieldIndex(idBroadcast.value)),
      row.getInt(row.fieldIndex(labelBroadcast.value)),
      StatObject.extractValues(row, featuresBroadcast.value)))
  }


  def defineTestLabelValues(spark: SparkSession, data: DataFrame): RDD[(String, Array[Double])] = {
    val idBroadcast = spark.sparkContext.broadcast(idColumn)
    val featuresBroadcast = spark.sparkContext.broadcast(featuresColumn)
    data.rdd.map(row => (row.getString(row.fieldIndex(idBroadcast.value)), StatObject.extractValues(row, featuresBroadcast.value)))

  }

  def defineLabelFeatures(spark: SparkSession, data: DataFrame, option: String): ChiSquareTask = {
    val defineFeatures = udf((p: mutable.WrappedArray[Double]) => Vectors.dense(p.toArray[Double]))
    if(option == "train"){
      val rdd = defineTrainLabelValues(spark, data)
      trainLabelFeatures = spark.createDataFrame(rdd).toDF(idColumn, labelColumn, "values")
        .withColumn(featureColumn, defineFeatures(col("values")))
        .select(idColumn, labelColumn, featureColumn)
    } else if(option == "test"){
      val rdd = defineTestLabelValues(spark, data)
      testLabelFeatures = spark.createDataFrame(rdd).toDF(idColumn, "values")
        .withColumn(featureColumn, defineFeatures(col("values")))
        .select(idColumn, featureColumn)
    }
    this
  }

  def defineChiSquareSelector(): ChiSqSelector = {
    new ChiSqSelector().setSelectorType("fpr").setFpr(alpha).setLabelCol(labelColumn).setFeaturesCol(featureColumn).setOutputCol(s"$featureColumn-Selected")
  }

  def fit(data: DataFrame): ChiSquareTask = {
    model = defineChiSquareSelector.fit(trainLabelFeatures)
    this
  }


  def transform(data: DataFrame, option: String): DataFrame = {
    if(option == "train"){model.transform(data).select(col(idColumn) , col(labelColumn), col(s"$featureColumn-Selected").alias(featureColumn))}
    else {model.transform(data).select(col(idColumn), col(s"$featureColumn-Selected").alias(featureColumn))}
  }

  def getLabelFeatures(option: String): DataFrame = {
    if(option == "train") {transform(trainLabelFeatures, option)} else {transform(testLabelFeatures, option)}
  }




}
