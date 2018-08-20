package fr.poverty.spark.classification.validation.crossValidation

import fr.poverty.spark.classification.gridParameters.GridParametersDecisionTree
import fr.poverty.spark.classification.task.DecisionTreeTask
import fr.poverty.spark.classification.validation.ValidationModelFactory
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.sql.DataFrame


class CrossValidationDecisionTreeTask(override val labelColumn: String,
                                      override val featureColumn: String,
                                      override val predictionColumn: String,
                                      override val metricName: String,
                                      override val pathSave: String,
                                      override val numFolds: Integer)
  extends CrossValidationTask(labelColumn, featureColumn, predictionColumn, metricName, pathSave, numFolds)
    with ValidationModelFactory {

  var estimator: DecisionTreeClassifier = _

  override def run(data: DataFrame): CrossValidationDecisionTreeTask = {
    defineEstimator()
    defineGridParameters()
    defineEvaluator()
    defineValidatorModel()
    fit(data)
    this
  }

  override def defineEstimator(): CrossValidationDecisionTreeTask = {
    estimator = new DecisionTreeTask(labelColumn=labelColumn, featureColumn=featureColumn, predictionColumn=predictionColumn).defineModel.getModel
    this
  }

  override def defineGridParameters(): CrossValidationDecisionTreeTask = {
    paramGrid = GridParametersDecisionTree.getParamsGrid(estimator)
    this
  }

  override def defineValidatorModel(): CrossValidationDecisionTreeTask = {
    crossValidator = new CrossValidator().setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setEstimator(estimator).setNumFolds(numFolds)
    this
  }

  def getEstimator: DecisionTreeClassifier = estimator

  def getBestModel: DecisionTreeClassificationModel = crossValidatorModel.bestModel.asInstanceOf[DecisionTreeClassificationModel]

}