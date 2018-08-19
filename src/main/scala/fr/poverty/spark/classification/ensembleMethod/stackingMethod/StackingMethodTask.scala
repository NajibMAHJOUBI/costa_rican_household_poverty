package fr.poverty.spark.classification.ensembleMethod.stackingMethod

import fr.poverty.spark.utils.{IndexToStringTask, LoadDataSetTask, StringIndexerTask}
import org.apache.spark.ml.feature.StringIndexerModel
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

import scala.collection.mutable

class StackingMethodTask(val idColumn: String, val labelColumn: String, val predictionColumn: String,
                         val pathPrediction: List[String], val mapFormat: Map[String, String],
                         val pathTrain: String, val formatTrain: String,
                         val pathStringIndexer: String, val pathSave: String,
                         val validationMethod: String, val ratio: Double) {

  var data: DataFrame = _
  var prediction: DataFrame = _
  var labelFeatures: DataFrame = _
  var predictionLabelFeatures: DataFrame = _
  var submissionLabelFeatures: DataFrame = _
  var stringIndexerModel: StringIndexerModel = _
  var transformPrediction: DataFrame = _
  var transformSubmission: DataFrame = _

  def getData: DataFrame = data

  def getPrediction: DataFrame = prediction

  def getPredictionLabelFeatures: DataFrame = predictionLabelFeatures

  def getSubmissionLabelFeatures: DataFrame = submissionLabelFeatures

  def getTransformPrediction: DataFrame =  transformPrediction

  def getTransformSubmission: DataFrame = transformSubmission

  def mergeData(spark: SparkSession, option: String): StackingMethodTask = {
    loadDataLabel(spark, option)
    pathPrediction.foreach(path => data = data.join(loadDataPredictionByLabel(spark, path, pathPrediction.indexOf(path), option), Seq(idColumn)))
    this
  }

  def loadDataPredictionByLabel(spark: SparkSession, path: String, index: Int, option: String): DataFrame = {
    val data = new LoadDataSetTask(path, format = mapFormat(option)).run(spark, option).select(col(idColumn), col(labelColumn))
    stringIndexerModel.transform(data).withColumnRenamed(stringIndexerModel.getOutputCol, s"prediction_${index.toString}").drop(stringIndexerModel.getInputCol)
  }

  def loadDataLabel(spark: SparkSession, option: String): StackingMethodTask = {
    loadStringIndexerModel()
    if(option == "prediction") {
      data = new LoadDataSetTask(sourcePath = pathTrain, format = formatTrain).run(spark, "train").select(col(idColumn), col(labelColumn))
      data = stringIndexerModel.transform(data).drop(labelColumn)
    } else if(option == "submission") {
      data = new LoadDataSetTask(sourcePath = pathTrain, format = formatTrain).run(spark, "test").select(col(idColumn))
    }
    this
  }

  def createLabelFeatures(spark: SparkSession, option: String): DataFrame = {
    mergeData(spark, option)
    val idColumnBroadcast = spark.sparkContext.broadcast(idColumn)
    val classificationMethodsBroadcast = spark.sparkContext.broadcast(pathPrediction.toArray)
    val features = (p: Row) => {StackingMethodObject.extractValues(p, classificationMethodsBroadcast.value)}
    val defineFeatures = udf((p: mutable.WrappedArray[Double]) => Vectors.dense(p.toArray[Double]))
    if(option == "prediction"){
      val rdd = data.rdd.map(p => (p.getString(p.fieldIndex(idColumnBroadcast.value)), p.getDouble(p.fieldIndex("label")), features(p)))
      labelFeatures = spark.createDataFrame(rdd).toDF(idColumn, labelColumn, "values")
        .withColumn("features", defineFeatures(col("values")))
        .select(idColumn, labelColumn, "features")
    } else if(option == "submission"){
      val rdd = data.rdd.map(p => (p.getString(p.fieldIndex(idColumnBroadcast.value)), features(p)))
      labelFeatures = spark.createDataFrame(rdd).toDF(idColumn, "values")
        .withColumn("features", defineFeatures(col("values")))
        .select(idColumn, "features")
    }
    labelFeatures
  }

  def loadStringIndexerModel(): StackingMethodTask = {
    stringIndexerModel = new StringIndexerTask("", "", "").loadModel(pathStringIndexer)
    this
  }

  def getIndexToString(): IndexToStringTask = {
    new IndexToStringTask("prediction", labelColumn, stringIndexerModel.labels)
  }

  def savePrediction(): StackingMethodTask = {
    getIndexToString().run(transformPrediction)
      .select(col(idColumn), col(labelColumn).cast(IntegerType))
      .write
      .mode("overwrite")
      .parquet(s"$pathSave/prediction")
    this
  }

  def saveSubmission(): StackingMethodTask = {
    getIndexToString().run(transformSubmission)
      .select(col(idColumn), col(labelColumn).cast(IntegerType))
      .repartition(1)
      .write
      .option("header", "true")
      .option("delimiter", ",")
      .mode("overwrite")
      .csv(s"$pathSave/submission")
    this
  }

}
