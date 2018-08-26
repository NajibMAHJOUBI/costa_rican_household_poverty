package fr.poverty.spark.classification.validation.trainValidation

import fr.poverty.spark.classification.gridParameters.GridParametersGbtClassifier
import fr.poverty.spark.classification.task.GbtClassifierTask
import fr.poverty.spark.classification.validation.ValidationModelFactory
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.tuning.TrainValidationSplit
import org.apache.spark.sql.DataFrame


class TrainValidationGbtClassifierTask(override val labelColumn: String,
                                       override val featureColumn: String,
                                       override val predictionColumn: String,
                                       override val metricName: String,
                                       override val pathSave: String,
                                       override val trainRatio: Double)
  extends TrainValidationTask(labelColumn, featureColumn, predictionColumn, metricName, pathSave, trainRatio)
    with ValidationModelFactory {

  override def run(data: DataFrame): TrainValidationGbtClassifierTask = {
    defineEstimator()
    defineGridParameters()
    defineEvaluator()
    defineValidatorModel()
    fit(data)
    this
  }

  override def defineEstimator(): TrainValidationGbtClassifierTask = {
    estimator = new GbtClassifierTask(labelColumn = labelColumn, featureColumn = featureColumn, predictionColumn = predictionColumn).defineEstimator.getEstimator
    this
  }

  override def defineGridParameters(): TrainValidationGbtClassifierTask = {
    paramGrid = GridParametersGbtClassifier.getParamsGrid(estimator.asInstanceOf[GBTClassifier])
    this
  }

  override def defineValidatorModel(): TrainValidationGbtClassifierTask = {
    trainValidator = new TrainValidationSplit().setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setEstimator(estimator).setTrainRatio(trainRatio)
    this
  }

}