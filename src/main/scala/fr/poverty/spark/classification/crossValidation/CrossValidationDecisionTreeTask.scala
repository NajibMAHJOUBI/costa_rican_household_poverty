package fr.poverty.spark.classification.crossValidation

import fr.poverty.spark.classification.gridParameters.GridParametersDecisionTree
import fr.poverty.spark.classification.task.{CrossValidationModelFactory, DecisionTreeTask}
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.DataFrame


class CrossValidationDecisionTreeTask(override val labelColumn: String,
                                      override val featureColumn: String,
                                      override val predictionColumn: String,
                                      override val numFolds: Integer,
                                      override val pathSave: String) extends CrossValidationTask(labelColumn, featureColumn, predictionColumn, numFolds, pathSave) with CrossValidationModelFactory {

  var estimator: DecisionTreeClassifier = _

  override def run(data: DataFrame): CrossValidationDecisionTreeTask = {
    defineEstimator()
    defineGridParameters()
    defineEvaluator()
    defineCrossValidatorModel()
    fit(data)
    this
  }

  override def defineEstimator(): CrossValidationDecisionTreeTask = {
    estimator = new DecisionTreeTask(labelColumn=labelColumn,
                                     featureColumn=featureColumn,
                                     predictionColumn=predictionColumn).defineModel.getModel
    this
  }

  override def defineGridParameters(): CrossValidationDecisionTreeTask = {
      paramGrid = new ParamGridBuilder()
        .addGrid(estimator.maxDepth, GridParametersDecisionTree.getMaxDepth)
        .addGrid(estimator.maxBins, GridParametersDecisionTree.getMaxBins)
        .build()
    this
  }

  override def defineCrossValidatorModel(): CrossValidationDecisionTreeTask = {
    crossValidator = new CrossValidator()
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setEstimator(estimator)
      .setNumFolds(numFolds)
    this
  }

  def getEstimator: DecisionTreeClassifier = estimator

  def getBestModel: DecisionTreeClassificationModel = crossValidatorModel.bestModel.asInstanceOf[DecisionTreeClassificationModel]

}