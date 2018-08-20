package fr.poverty.spark.classification.validation.crossValidation

import fr.poverty.spark.classification.gridParameters.GridParametersRandomForest
import fr.poverty.spark.classification.task.RandomForestTask
import fr.poverty.spark.classification.validation.ValidationModelFactory
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.sql.DataFrame


class CrossValidationRandomForestTask(override val labelColumn: String,
                                      override val featureColumn: String,
                                      override val predictionColumn: String,
                                      override val metricName: String,
                                      override val pathSave: String,
                                      override val numFolds: Integer)
  extends CrossValidationTask(labelColumn, featureColumn, predictionColumn, metricName, pathSave, numFolds)
    with ValidationModelFactory {

  var estimator: RandomForestClassifier = _

  override def run(data: DataFrame): CrossValidationRandomForestTask = {
    defineEstimator()
    defineGridParameters()
    defineEvaluator()
    defineValidatorModel()
    fit(data)
    this
  }

  override def defineEstimator(): CrossValidationRandomForestTask = {
    estimator = new RandomForestTask(labelColumn=labelColumn, featureColumn=featureColumn, predictionColumn=predictionColumn).defineModel.getModel
    this
  }

  override def defineGridParameters(): CrossValidationRandomForestTask = {
    paramGrid = GridParametersRandomForest.getParamsGrid(estimator)
    this
  }

  override def defineValidatorModel(): CrossValidationRandomForestTask = {
    crossValidator = new CrossValidator().setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setEstimator(estimator).setNumFolds(numFolds)
    this
  }

  def getEstimator: RandomForestClassifier = estimator

  def getBestModel: RandomForestClassificationModel = crossValidatorModel.bestModel.asInstanceOf[RandomForestClassificationModel]

}