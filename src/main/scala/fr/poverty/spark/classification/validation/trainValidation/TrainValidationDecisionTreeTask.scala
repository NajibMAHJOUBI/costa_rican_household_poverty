package fr.poverty.spark.classification.validation.trainValidation

import fr.poverty.spark.classification.gridParameters.GridParametersDecisionTree
import fr.poverty.spark.classification.task.DecisionTreeTask
import fr.poverty.spark.classification.validation.ValidationModelFactory
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.tuning.TrainValidationSplit
import org.apache.spark.sql.DataFrame


class TrainValidationDecisionTreeTask(override val labelColumn: String,
                                      override val featureColumn: String,
                                      override val predictionColumn: String,
                                      override val metricName: String,
                                      override val pathSave: String,
                                      override val trainRatio: Double)
  extends TrainValidationTask(labelColumn, featureColumn, predictionColumn, metricName, pathSave, trainRatio)
    with ValidationModelFactory {

  var estimator: DecisionTreeClassifier = _

  override def run(data: DataFrame): TrainValidationDecisionTreeTask = {
    defineEstimator()
    defineGridParameters()
    defineEvaluator()
    defineValidatorModel()
    fit(data)
    this
  }

  override def defineEstimator(): TrainValidationDecisionTreeTask = {
    estimator = new DecisionTreeTask(labelColumn = labelColumn, featureColumn = featureColumn, predictionColumn = predictionColumn).defineModel.getModel
    this
  }

  override def defineGridParameters(): TrainValidationDecisionTreeTask = {
    paramGrid = GridParametersDecisionTree.getParamsGrid(estimator)
    this
  }

  override def defineValidatorModel(): TrainValidationDecisionTreeTask = {
    trainValidator = new TrainValidationSplit().setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setEstimator(estimator).setTrainRatio(trainRatio)
    this
  }

  def getEstimator: DecisionTreeClassifier = estimator

  def getBestModel: DecisionTreeClassificationModel = trainValidatorModel.bestModel.asInstanceOf[DecisionTreeClassificationModel]

}