package fr.poverty.spark.classification.validation.crossValidation

import fr.poverty.spark.classification.gridParameters.GridParametersLogisticRegression
import fr.poverty.spark.classification.task.LogisticRegressionTask
import fr.poverty.spark.classification.validation.ValidationModelFactory
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.DataFrame


class CrossValidationLogisticRegressionTask(override val labelColumn: String,
                                            override val featureColumn: String,
                                            override val predictionColumn: String,
                                            override val pathSave: String,
                                            override val numFolds: Integer) extends CrossValidationTask(labelColumn, featureColumn, predictionColumn, pathSave, numFolds) with ValidationModelFactory {

  var estimator: LogisticRegression = _

  override def run(data: DataFrame): CrossValidationLogisticRegressionTask = {
    defineEstimator()
    defineGridParameters()
    defineEvaluator()
    defineValidatorModel()
    fit(data)
    this
  }

  override def defineEstimator(): CrossValidationLogisticRegressionTask = {
    estimator = new LogisticRegressionTask(labelColumn=labelColumn,
                                           featureColumn=featureColumn,
                                           predictionColumn=predictionColumn).defineModel.getModel
    this
  }

  override def defineGridParameters(): CrossValidationLogisticRegressionTask = {
    paramGrid = GridParametersLogisticRegression.getParamsGrid(estimator)
    this
  }

  override def defineValidatorModel(): CrossValidationLogisticRegressionTask = {
    crossValidator = new CrossValidator()
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setEstimator(estimator)
      .setNumFolds(numFolds)
    this
  }

  def getEstimator: LogisticRegression = estimator

  def getBestModel: LogisticRegressionModel = crossValidatorModel.bestModel.asInstanceOf[LogisticRegressionModel]

}