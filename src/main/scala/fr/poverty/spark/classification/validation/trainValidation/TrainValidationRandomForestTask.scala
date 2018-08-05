package fr.poverty.spark.classification.validation.trainValidation

import fr.poverty.spark.classification.gridParameters.GridParametersRandomForest
import fr.poverty.spark.classification.task.RandomForestTask
import fr.poverty.spark.classification.validation.ValidationModelFactory
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.tuning.TrainValidationSplit
import org.apache.spark.sql.DataFrame


class TrainValidationRandomForestTask(override val labelColumn: String,
                                      override val featureColumn: String,
                                      override val predictionColumn: String,
                                      override val pathSave: String,
                                      override val trainRatio: Double)
  extends TrainValidationTask(labelColumn, featureColumn, predictionColumn, pathSave,
    trainRatio) with ValidationModelFactory {

  var estimator: RandomForestClassifier = _

  override def run(data: DataFrame): TrainValidationRandomForestTask = {
    defineEstimator()
    defineGridParameters()
    defineEvaluator()
    defineValidatorModel()
    fit(data)
    this
  }

  override def defineEstimator(): TrainValidationRandomForestTask = {
    estimator = new RandomForestTask(labelColumn = labelColumn, featureColumn = featureColumn, predictionColumn = predictionColumn).defineModel.getModel
    this
  }

  override def defineGridParameters(): TrainValidationRandomForestTask = {
    paramGrid = GridParametersRandomForest.getParamsGrid(estimator)
    this
  }

  override def defineValidatorModel(): TrainValidationRandomForestTask = {
    trainValidator = new TrainValidationSplit().setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setEstimator(estimator).setTrainRatio(trainRatio)
    this
  }

  def getEstimator: RandomForestClassifier = estimator

  def getBestModel: RandomForestClassificationModel = trainValidatorModel.bestModel.asInstanceOf[RandomForestClassificationModel]

}