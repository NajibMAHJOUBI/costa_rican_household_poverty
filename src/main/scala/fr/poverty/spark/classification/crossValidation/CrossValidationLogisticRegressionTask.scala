package fr.poverty.spark.classification.crossValidation

import fr.poverty.spark.classification.gridParameters.GridParametersLogisticRegression
import fr.poverty.spark.classification.task.{CrossValidationModelFactory, LogisticRegressionTask}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.DataFrame


class CrossValidationLogisticRegressionTask(override val labelColumn: String,
                                            override val featureColumn: String,
                                            override val predictionColumn: String,
                                            override val pathSave: String) extends CrossValidationTask(labelColumn, featureColumn, predictionColumn, pathSave) with CrossValidationModelFactory {

  var estimator: LogisticRegression = _

  override def run(data: DataFrame): CrossValidationLogisticRegressionTask = {
    defineEstimator()
    defineGridParameters()
    defineEvaluator()
    defineCrossValidatorModel()
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
      paramGrid = new ParamGridBuilder()
        .addGrid(estimator.regParam, GridParametersLogisticRegression.getRegParam)
        .addGrid(estimator.elasticNetParam, GridParametersLogisticRegression.getElasticNetParam)
        .build()
    this
  }

  override def defineCrossValidatorModel(): CrossValidationLogisticRegressionTask = {
    crossValidator = new CrossValidator()
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setEstimator(estimator)
    this
  }

  def getEstimator: LogisticRegression = estimator

  def getBestModel: LogisticRegressionModel = crossValidatorModel.bestModel.asInstanceOf[LogisticRegressionModel]

}