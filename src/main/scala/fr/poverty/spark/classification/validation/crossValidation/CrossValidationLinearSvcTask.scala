package fr.poverty.spark.classification.validation.crossValidation

import fr.poverty.spark.classification.gridParameters.GridParametersLinearSvc
import fr.poverty.spark.classification.task.LinearSvcTask
import fr.poverty.spark.classification.validation.ValidationModelFactory
import org.apache.spark.ml.classification.{LinearSVC, LinearSVCModel}
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.sql.DataFrame


class CrossValidationLinearSvcTask(override val labelColumn: String,
                                   override val featureColumn: String,
                                   override val predictionColumn: String,
                                   override val metricName: String,
                                   override val pathSave: String,
                                   override val numFolds: Integer) extends
  CrossValidationTask(labelColumn, featureColumn, predictionColumn, metricName, pathSave, numFolds) with ValidationModelFactory {

  var estimator: LinearSVC = _

  override def run(data: DataFrame): CrossValidationLinearSvcTask = {
    defineEstimator()
    defineGridParameters()
    defineEvaluator()
    defineValidatorModel()
    fit(data)
    this
  }

  override def defineEstimator(): CrossValidationLinearSvcTask = {
    estimator = new LinearSvcTask(labelColumn=labelColumn, featureColumn=featureColumn, predictionColumn=predictionColumn).defineModel.getModel
    this
  }

  def defineGridParameters(): CrossValidationLinearSvcTask = {
      paramGrid = GridParametersLinearSvc.getParamsGrid(estimator)
    this
  }

  def defineValidatorModel(): CrossValidationLinearSvcTask = {
    crossValidator = new CrossValidator().setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setEstimator(estimator).setNumFolds(numFolds)
    this
  }

  def getEstimator: LinearSVC = estimator

  def getBestModel: LinearSVCModel = crossValidatorModel.bestModel.asInstanceOf[LinearSVCModel]

}