package fr.poverty.spark.classification.validation.crossValidation

import fr.poverty.spark.classification.gridParameters.GridParametersGbtClassifier
import fr.poverty.spark.classification.task.GbtClassifierTask
import fr.poverty.spark.classification.validation.ValidationModelFactory
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.sql.DataFrame


class CrossValidationGbtClassifierTask(override val labelColumn: String,
                                       override val featureColumn: String,
                                       override val predictionColumn: String,
                                       override val metricName: String,
                                       override val pathSave: String,
                                       override val numFolds: Integer)
  extends CrossValidationTask(labelColumn, featureColumn, predictionColumn, metricName, pathSave, numFolds)
    with ValidationModelFactory {

  override def run(data: DataFrame): CrossValidationGbtClassifierTask = {
    defineEstimator()
    defineGridParameters()
    defineEvaluator()
    defineValidatorModel()
    fit(data)
    this
  }

  override def defineEstimator(): CrossValidationGbtClassifierTask = {
    estimator = new GbtClassifierTask(labelColumn=labelColumn, featureColumn=featureColumn, predictionColumn=predictionColumn).defineEstimator.getEstimator
    this
  }

  override def defineGridParameters(): CrossValidationGbtClassifierTask = {
    paramGrid = GridParametersGbtClassifier.getParamsGrid(estimator.asInstanceOf[GBTClassifier])
    this
  }

  override def defineValidatorModel(): CrossValidationGbtClassifierTask = {
    crossValidator = new CrossValidator().setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setEstimator(estimator).setNumFolds(numFolds)
    this
  }

}