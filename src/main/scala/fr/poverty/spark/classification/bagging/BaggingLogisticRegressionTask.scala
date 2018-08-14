package fr.poverty.spark.classification.bagging

import fr.poverty.spark.classification.validation.crossValidation.CrossValidationLogisticRegressionTask
import fr.poverty.spark.classification.validation.trainValidation.TrainValidationLogisticRegressionTask
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{DataFrame, SparkSession}

class BaggingLogisticRegressionTask(override val idColumn: String, override val labelColumn: String,
                                    override val featureColumn: String, override val predictionColumn: String,
                                    override val pathSave: String,
                                    override val numberOfSampling: Int, override val samplingFraction: Double,
                                    override val validationMethod: String, override val ratio: Double) extends
  BaggingTask(idColumn, labelColumn, featureColumn, predictionColumn, pathSave, numberOfSampling,
    samplingFraction, validationMethod, ratio) {

  var modelFittedList: List[LogisticRegressionModel] = List()

  def run(data: DataFrame): Unit = {
    defineSampleSubset(data)
    loopDataSampling()
  }

  def loopDataSampling(): Unit = {
    sampleSubsetsList.foreach(sample => {
      if(validationMethod == "trainValidation"){
        val trainValidation = new TrainValidationLogisticRegressionTask(labelColumn, featureColumn, predictionColumn, "", ratio)
        trainValidation.run(sample)
        modelFittedList = modelFittedList ++ List(trainValidation.getBestModel)
      } else if(validationMethod == "crossValidation"){
        val crossValidation = new CrossValidationLogisticRegressionTask(labelColumn, featureColumn, predictionColumn, "", ratio.toInt)
        crossValidation.run(sample)
        modelFittedList = modelFittedList ++ List(crossValidation.getBestModel)
      }
    })
  }

  def computePrediction(spark: SparkSession, data: DataFrame): DataFrame = {
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
      mergePredictions(p, numberOfModels.value)))
    spark.createDataFrame(rdd).toDF(idColumn, labelColumn, predictionColumn)
  }

  def computeSubmission(data: DataFrame): Unit = {}

}
