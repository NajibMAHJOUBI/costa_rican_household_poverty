package fr.poverty.spark.classification.crossValidation

import fr.poverty.spark.classification.gridParameters.GridParametersLinearSvc
import fr.poverty.spark.classification.task.{CrossValidationModelFactory, LinearSvcTask}
import org.apache.spark.ml.classification.{LinearSVC, LinearSVCModel}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.DataFrame


class CrossValidationLinearSvcTask(override val labelColumn: String,
                                   override val featureColumn: String,
                                   override val predictionColumn: String,
                                   override val pathSave: String) extends CrossValidationTask(labelColumn, featureColumn, predictionColumn, pathSave) with CrossValidationModelFactory {

  var estimator: LinearSVC = _

  override def run(data: DataFrame): CrossValidationLinearSvcTask = {
    defineEstimator()
    defineGridParameters()
    defineEvaluator()
    defineCrossValidatorModel()
    fit(data)
    saveModel()
    this
  }

  override def defineEstimator(): CrossValidationLinearSvcTask = {
    estimator = new LinearSvcTask(labelColumn=labelColumn,
                                  featureColumn=featureColumn,
                                  predictionColumn=predictionColumn).defineModel.getModel
    this
  }

  def defineGridParameters(): CrossValidationLinearSvcTask = {
      paramGrid = new ParamGridBuilder()
        .addGrid(estimator.regParam, GridParametersLinearSvc.getRegParam)
        .addGrid(estimator.fitIntercept, GridParametersLinearSvc.getFitIntercept)
        .addGrid(estimator.standardization, GridParametersLinearSvc.getStandardization)
        .build()
    this
  }

  def defineCrossValidatorModel(): CrossValidationLinearSvcTask = {
    crossValidator = new CrossValidator()
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setEstimator(estimator)
    this
  }

  def getEstimator: LinearSVC = estimator

  def getBestModel: LinearSVCModel = crossValidatorModel.bestModel.asInstanceOf[LinearSVCModel]

}