package fr.poverty.spark.classification.validation.trainValidation

import fr.poverty.spark.classification.gridParameters.GridParametersGbtClassifier
import fr.poverty.spark.classification.task.GbtClassifierTask
import fr.poverty.spark.classification.validation.ValidationModelFactory
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.tuning.TrainValidationSplit
import org.apache.spark.sql.DataFrame


class TrainValidationGbtClassifierTask(override val labelColumn: String,
                                       override val featureColumn: String,
                                       override val predictionColumn: String,
                                       override val pathSave: String,
                                       override val trainRatio: Double) extends
  TrainValidationTask(labelColumn, featureColumn, predictionColumn,
    pathSave, trainRatio) with ValidationModelFactory {

  var estimator: GBTClassifier = _

  override def run(data: DataFrame): TrainValidationGbtClassifierTask = {
    defineEstimator()
    defineGridParameters()
    defineEvaluator()
    defineValidatorModel()
    fit(data)
    this
  }

  override def defineEstimator(): TrainValidationGbtClassifierTask = {
    estimator = new GbtClassifierTask(labelColumn = labelColumn, featureColumn = featureColumn, predictionColumn = predictionColumn).defineModel.getModel
    this
  }

  override def defineGridParameters(): TrainValidationGbtClassifierTask = {
    paramGrid = GridParametersGbtClassifier.getParamsGrid(estimator)
    this
  }

  override def defineValidatorModel(): TrainValidationGbtClassifierTask = {
    trainValidator = new TrainValidationSplit().setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setEstimator(estimator).setTrainRatio(trainRatio)
    this
  }

  def getEstimator: GBTClassifier = estimator

  def getBestModel: GBTClassificationModel = trainValidatorModel.bestModel.asInstanceOf[GBTClassificationModel]

}