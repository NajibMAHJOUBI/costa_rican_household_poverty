package fr.poverty.spark.classification.validation.trainValidation

import fr.poverty.spark.classification.validation.ValidationTask
import org.apache.spark.ml.tuning.{TrainValidationSplit, TrainValidationSplitModel}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.DataFrame

class TrainValidationTask(override val labelColumn: String,
                          override val featureColumn: String,
                          override val predictionColumn: String,
                          override val metricName: String,
                          override val pathSave: String,
                          val trainRatio: Double) extends
  ValidationTask(labelColumn, featureColumn, predictionColumn, metricName, pathSave) {

  var trainValidator: TrainValidationSplit = _
  var trainValidatorModel: TrainValidationSplitModel = _
  var estimator: Estimator[_] = _

  def fit(data: DataFrame): TrainValidationTask = {
    trainValidatorModel = trainValidator.fit(data)
    this
  }

  def transform(data: DataFrame): TrainValidationTask = {
    prediction = trainValidatorModel.transform(data)
    this
  }

  def saveModel(): Unit = {
    trainValidatorModel.write.overwrite().save(s"$pathSave/model")
  }

  def getTrainValidator: TrainValidationSplit = trainValidator

  def getTrainValidatorModel: TrainValidationSplitModel = trainValidatorModel

  def getEstimator: Estimator[_] = estimator

  def getBestModel: Model[_] = trainValidatorModel.bestModel

}
