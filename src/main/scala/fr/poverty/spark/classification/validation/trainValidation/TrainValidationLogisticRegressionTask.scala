package fr.poverty.spark.classification.validation.trainValidation


import fr.poverty.spark.classification.gridParameters.GridParametersLogisticRegression
import fr.poverty.spark.classification.task.LogisticRegressionTask
import fr.poverty.spark.classification.validation.ValidationModelFactory
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.tuning.TrainValidationSplit
import org.apache.spark.sql.DataFrame


class TrainValidationLogisticRegressionTask(override val labelColumn: String,
                                            override val featureColumn: String,
                                            override val predictionColumn: String,
                                            override val metricName: String,
                                            override val pathSave: String,
                                            override val trainRatio: Double)
  extends TrainValidationTask(labelColumn, featureColumn, predictionColumn, metricName, pathSave, trainRatio)
    with ValidationModelFactory {

  override def run(data: DataFrame): TrainValidationLogisticRegressionTask = {
    defineEstimator()
    defineGridParameters()
    defineEvaluator()
    defineValidatorModel()
    fit(data)
    this
  }

  override def defineEstimator(): TrainValidationLogisticRegressionTask = {
    estimator = new LogisticRegressionTask(labelColumn = labelColumn, featureColumn = featureColumn, predictionColumn = predictionColumn).defineEstimator.getEstimator
    this
  }

  override def defineGridParameters(): TrainValidationLogisticRegressionTask = {
    paramGrid = GridParametersLogisticRegression.getParamsGrid(estimator.asInstanceOf[LogisticRegression])
    this
  }

  override def defineValidatorModel(): TrainValidationLogisticRegressionTask = {
    trainValidator = new TrainValidationSplit().setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setEstimator(estimator).setTrainRatio(trainRatio)
    this
  }

}