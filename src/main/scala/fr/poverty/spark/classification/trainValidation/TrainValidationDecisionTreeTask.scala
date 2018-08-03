package fr.poverty.spark.classification.trainValidation

import fr.poverty.spark.classification.gridParameters.GridParametersDecisionTree
import fr.poverty.spark.classification.task.DecisionTreeTask
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.tuning.TrainValidationSplit
import org.apache.spark.sql.DataFrame


class TrainValidationDecisionTreeTask(override val labelColumn: String, override val featureColumn: String, override val predictionColumn: String, override val trainRatio: Double, override val pathSave: String) extends TrainValidationTask(labelColumn, featureColumn, predictionColumn, trainRatio, pathSave) with TrainValidationModelFactory {

  var estimator: DecisionTreeClassifier = _

  override def run(data: DataFrame): TrainValidationDecisionTreeTask = {
    defineEstimator()
    defineGridParameters()
    defineEvaluator()
    defineTrainValidatorModel()
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

  override def defineTrainValidatorModel(): TrainValidationDecisionTreeTask = {
    trainValidator = new TrainValidationSplit().setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setEstimator(estimator).setTrainRatio(trainRatio)
    this
  }

  def getEstimator: DecisionTreeClassifier = {
    estimator
  }

  def getBestModel: DecisionTreeClassificationModel = {
    trainValidatorModel.bestModel.asInstanceOf[DecisionTreeClassificationModel]
  }

}