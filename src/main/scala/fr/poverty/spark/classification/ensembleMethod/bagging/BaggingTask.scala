package fr.poverty.spark.classification.ensembleMethod.bagging

import fr.poverty.spark.classification.ensembleMethod.adaBoosting.AdaBoostingTask
import org.apache.spark.ml.Model
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

class BaggingTask(val idColumn: String, val labelColumn: String, val featureColumn: String,
                  val predictionColumn: String, val pathSave: String,
                  val numberOfSampling: Int, val samplingFraction: Double,
                  val validationMethod: String, val ratio: Double) {

  var sampleSubsetsList: List[DataFrame] = List()

  def defineSampleSubset(data: DataFrame): BaggingTask = {
    (1 to numberOfSampling).foreach(index => {
      sampleSubsetsList = sampleSubsetsList ++ List(data.sample(true, samplingFraction))
    })
    this
  }

  def computePrediction(spark: SparkSession, data: DataFrame, modelFittedList: List[Model[_]]): DataFrame = {
    val idColumnBroadcast = spark.sparkContext.broadcast(idColumn)
    val labelColumnBroadcast = spark.sparkContext.broadcast(labelColumn)
    val numberOfModels = spark.sparkContext.broadcast(modelFittedList.length)
    var idDataSet: DataFrame = data.select(idColumn, labelColumn)
    modelFittedList.foreach(model => {
      val weakTransform = model.transform(data).select(idColumn, predictionColumn)
        .select(col(idColumn), col("prediction").alias(s"prediction_${modelFittedList.indexOf(model)}"))
      idDataSet = idDataSet.join(weakTransform, Seq(idColumn))
    })
    val rdd = idDataSet.rdd.map(p => (p.getString(p.fieldIndex(idColumnBroadcast.value)),
      p.getDouble(p.fieldIndex(labelColumnBroadcast.value)),
      BaggingObject.mergePredictions(p, numberOfModels.value)))
    spark.createDataFrame(rdd).toDF(idColumn, labelColumn, predictionColumn)
  }

  def computeSubmission(spark: SparkSession, data: DataFrame, modelFittedList: List[Model[_]]): DataFrame = {
    val idColumnBroadcast = spark.sparkContext.broadcast(idColumn)
    val numberOfModels = spark.sparkContext.broadcast(modelFittedList.length)
    var idDataSet: DataFrame = data.select(idColumn)
    modelFittedList.foreach(model => {
      val weakTransform = model.transform(data).select(idColumn, predictionColumn)
        .select(col(idColumn), col("prediction").alias(s"prediction_${modelFittedList.indexOf(model)}"))
      idDataSet = idDataSet.join(weakTransform, Seq(idColumn))
    })
    val rdd = idDataSet.rdd.map(p => (p.getString(p.fieldIndex(idColumnBroadcast.value)),
      BaggingObject.mergePredictions(p, numberOfModels.value)))
    spark.createDataFrame(rdd).toDF(idColumn, predictionColumn)
  }

  def savePrediction(data: DataFrame): BaggingTask = {
    data
      .select(col(idColumn), col("Target").cast(IntegerType))
      .write
      .mode("overwrite")
      .parquet(s"$pathSave/prediction")
    this
  }

  def saveSubmission(data: DataFrame): BaggingTask = {
    data
      .select(col(idColumn), col("Target").cast(IntegerType))
      .repartition(1)
      .write
      .option("header", "true")
      .option("delimiter", ",")
      .mode("overwrite")
      .csv(s"$pathSave/submission")
    this
  }
}
