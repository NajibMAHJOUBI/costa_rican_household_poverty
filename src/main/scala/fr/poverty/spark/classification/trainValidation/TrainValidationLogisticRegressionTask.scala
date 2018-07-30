package fr.poverty.spark.classification.trainValidation


import fr.poverty.spark.classification.gridParameters.GridParametersLogisticRegression
import fr.poverty.spark.classification.task.LogisticRegressionTask
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.DataFrame


class TrainValidationLogisticRegressionTask(override val labelColumn: String, override val featureColumn: String, override val predictionColumn: String, override val trainRatio: Double, override val pathSave: String) extends TrainValidationTask(labelColumn, featureColumn, predictionColumn, trainRatio, pathSave) with TrainValidationModelFactory {

  var estimator: LogisticRegression = _

  override def run(data: DataFrame): TrainValidationLogisticRegressionTask = {
    defineEstimator()
    defineGridParameters()
    defineEvaluator()
    defineTrainValidatorModel()
    fit(data)
    saveModel()
    this
  }

  override def defineEstimator(): TrainValidationLogisticRegressionTask = {
    estimator = new LogisticRegressionTask(labelColumn = labelColumn, featureColumn = featureColumn, predictionColumn = predictionColumn).defineModel.getModel
    this
  }

  override def defineGridParameters(): TrainValidationLogisticRegressionTask = {
    paramGrid = GridParametersLogisticRegression.getParamsGrid(estimator)
    this
  }

  override def defineTrainValidatorModel(): TrainValidationLogisticRegressionTask = {
    trainValidator = new TrainValidationSplit().setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setEstimator(estimator).setTrainRatio(trainRatio)
    this
  }

  def getEstimator: LogisticRegression = {
    estimator
  }

  def getBestModel: LogisticRegressionModel = {
    trainValidatorModel.bestModel.asInstanceOf[LogisticRegressionModel]
  }

}