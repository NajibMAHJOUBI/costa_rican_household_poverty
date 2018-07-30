package fr.poverty.spark.classification.crossValidation

import fr.poverty.spark.classification.gridParameters.GridParametersGbtClassifier
import fr.poverty.spark.classification.task.{CrossValidationModelFactory, GbtClassifierTask}
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.DataFrame


class CrossValidationGbtClassifierTask(override val labelColumn: String,
                                       override val featureColumn: String,
                                       override val predictionColumn: String,
                                       override val numFolds: Integer,
                                       override val pathSave: String) extends CrossValidationTask(labelColumn, featureColumn, predictionColumn, numFolds, pathSave) with CrossValidationModelFactory {

  var estimator: GBTClassifier = _

  override def run(data: DataFrame): CrossValidationGbtClassifierTask = {
    defineEstimator()
    defineGridParameters()
    defineEvaluator()
    defineCrossValidatorModel()
    fit(data)
    saveModel()
    this
  }

  override def defineEstimator(): CrossValidationGbtClassifierTask = {
    estimator = new GbtClassifierTask(labelColumn=labelColumn,
                                      featureColumn=featureColumn,
                                      predictionColumn=predictionColumn).defineModel.getModel
    this
  }

  override def defineGridParameters(): CrossValidationGbtClassifierTask = {
      paramGrid = new ParamGridBuilder()
        .addGrid(estimator.maxDepth, GridParametersGbtClassifier.getMaxDepth)
        .addGrid(estimator.maxBins, GridParametersGbtClassifier.getMaxBins)
        .build()
    this
  }

  override def defineCrossValidatorModel(): CrossValidationGbtClassifierTask = {
    crossValidator = new CrossValidator()
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setEstimator(estimator)
      .setNumFolds(numFolds)
    this
  }

  def getEstimator: GBTClassifier = estimator

  def getBestModel: GBTClassificationModel = crossValidatorModel.bestModel.asInstanceOf[GBTClassificationModel]

}