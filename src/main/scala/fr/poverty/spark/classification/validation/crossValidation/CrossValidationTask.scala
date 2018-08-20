package fr.poverty.spark.classification.validation.crossValidation

import fr.poverty.spark.classification.validation.ValidationTask
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel}
import org.apache.spark.sql.DataFrame

class CrossValidationTask(override val labelColumn: String,
                          override val featureColumn: String,
                          override val predictionColumn: String,
                          override val metricName: String,
                          override val pathSave: String,
                          val numFolds: Integer)
  extends ValidationTask(labelColumn, featureColumn, predictionColumn, metricName, pathSave) {

  var crossValidator: CrossValidator = _
  var crossValidatorModel: CrossValidatorModel = _

  def fit(data: DataFrame): CrossValidationTask = {
    crossValidatorModel = crossValidator.fit(data)
    this
  }

  def transform(data: DataFrame): CrossValidationTask = {
    prediction = crossValidatorModel.transform(data)
    this
  }

  def saveModel(): Unit = {
    crossValidatorModel.write.overwrite().save(s"$pathSave/model")
  }

  def getCrossValidator: CrossValidator = crossValidator

  def getCrossValidatorModel: CrossValidatorModel = crossValidatorModel

  }
