package fr.poverty.spark.classification.stackingMethod

import fr.poverty.spark.utils.LoadDataSetTask
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

import scala.collection.mutable

class StackingMethodTask(val classificationMethods: Array[String],
                         val pathTrain: String,
                         val pathPrediction: String,
                         val pathSave: String,
                         val validationMethod: String,
                         val ratio: Double,
                         val idColumn: String,
                         val labelColumn: String,
                         val predictionColumn: String) {

  var data: DataFrame = _

  def getData: DataFrame = {
    data
  }

  def mergeData(spark: SparkSession, label: String): StackingMethodTask = {
    data = loadDataLabel(spark, label)
    classificationMethods.foreach(method => data = data.join(loadDataPredictionByLabel(spark, method, label), Seq("id")))
    data = data.drop("id")
    this
  }

  def loadDataPredictionByLabel(spark: SparkSession, method: String, label: String): DataFrame = {
    new LoadDataSetTask(s"$pathPrediction/$method", format = "csv")
      .run(spark, "prediction")
      .select(col("id"), col(s"prediction_$label").alias(s"prediction_$method"))
  }

  def loadDataLabel(spark: SparkSession, label: String): DataFrame = {
    new LoadDataSetTask(sourcePath = pathTrain, format="csv").run(spark, "train")
      .select(col("id"), col(label).alias("label"))
  }

  def createLabelFeatures(spark: SparkSession, label: String): DataFrame = {
    val classificationMethodsBroadcast = spark.sparkContext.broadcast(classificationMethods)
    val features = (p: Row) => {
      StackingMethodObject.extractValues(p, classificationMethodsBroadcast.value)
    }
    val rdd = data.rdd.map(p => (p.getLong(p.fieldIndex("label")), features(p)))
    val labelFeatures = spark.createDataFrame(rdd).toDF(label, "values")
    val defineFeatures = udf((p: mutable.WrappedArray[Double]) => Vectors.dense(p.toArray[Double]))
    labelFeatures.withColumn("features", defineFeatures(col("values"))).select(label, "features")
  }

}
