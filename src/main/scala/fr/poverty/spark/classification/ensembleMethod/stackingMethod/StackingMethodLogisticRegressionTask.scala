package fr.poverty.spark.classification.ensembleMethod.stackingMethod

import fr.poverty.spark.classification.validation.crossValidation.CrossValidationLogisticRegressionTask
import fr.poverty.spark.classification.validation.trainValidation.TrainValidationLogisticRegressionTask
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.sql.{DataFrame, SparkSession}

class StackingMethodLogisticRegressionTask(override val idColumn: String,
                                           override val labelColumn: String,
                                           override val predictionColumn: String,
                                           override val pathPrediction: List[String],
                                           override val mapFormat: Map[String, String],
                                           override val pathTrain: String,
                                           override val formatTrain: String,
                                           override val pathStringIndexer: String,
                                           override val pathSave: String,
                                           override val validationMethod: String,
                                           override val ratio: Double,
                                           override val metricName: String)
  extends StackingMethodTask(idColumn, labelColumn, predictionColumn, pathPrediction, mapFormat, pathTrain, formatTrain, pathStringIndexer, pathSave, validationMethod, ratio, metricName)
    with StackingMethodFactory {

  val featureColumn: String = "features"
  var model: LogisticRegressionModel = _

  override def run(spark: SparkSession): StackingMethodLogisticRegressionTask = {
    predictionLabelFeatures = createLabelFeatures(spark, "prediction")
    submissionLabelFeatures = createLabelFeatures(spark, "submission")
    defineValidationModel(predictionLabelFeatures)
    transform()
    savePrediction()
    saveSubmission()
    this
  }

  override def defineValidationModel(data: DataFrame): StackingMethodLogisticRegressionTask = {
    if (validationMethod == "crossValidation") {
      val cv = new CrossValidationLogisticRegressionTask(labelColumn, featureColumn, "prediction", metricName, pathSave, ratio.toInt)
      cv.run(data)
      model = cv.getBestModel
    } else if (validationMethod == "trainValidation") {
      val tv = new TrainValidationLogisticRegressionTask(labelColumn, featureColumn, "prediction", metricName, pathSave, ratio.toDouble)
      tv.run(data)
      model = tv.getBestModel
    }
    this
  }

  override def saveModel(path: String): StackingMethodLogisticRegressionTask = {
    model.write.overwrite().save(path)
    this
  }

  def loadModel(path: String): StackingMethodLogisticRegressionTask = {
    model = LogisticRegressionModel.load(path)
    this
  }

  override def transform(): StackingMethodLogisticRegressionTask = {
    transformPrediction = model.transform(predictionLabelFeatures).drop(labelColumn)
    transformSubmission = model.transform(submissionLabelFeatures)
    this
  }

}
