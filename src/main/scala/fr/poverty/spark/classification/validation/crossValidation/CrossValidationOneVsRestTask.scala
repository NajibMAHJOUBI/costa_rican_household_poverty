package fr.poverty.spark.classification.validation.crossValidation

import fr.poverty.spark.classification.gridParameters.GridParametersOneVsRest
import fr.poverty.spark.classification.task.OneVsRestTask
import fr.poverty.spark.classification.validation.ValidationModelFactory
import org.apache.spark.ml.classification.OneVsRest
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.sql.DataFrame


class CrossValidationOneVsRestTask(override val labelColumn: String,
                                   override val featureColumn: String,
                                   override val predictionColumn: String,
                                   override val metricName: String,
                                   override val pathSave: String,
                                   override val numFolds: Integer,
                                   val classifier: String,
                                   val bernoulliOption: Boolean = false)
  extends CrossValidationTask(labelColumn, featureColumn, predictionColumn, metricName, pathSave, numFolds)
    with ValidationModelFactory {

  override def run(data: DataFrame): CrossValidationOneVsRestTask = {
    defineEstimator()
    defineGridParameters()
    defineEvaluator()
    defineValidatorModel()
    fit(data)
    this
  }

  override def defineEstimator(): CrossValidationOneVsRestTask = {
    estimator = new OneVsRestTask(labelColumn, featureColumn, predictionColumn, classifier).defineEstimator.getEstimator
    this
  }

  override def defineGridParameters(): CrossValidationOneVsRestTask = {
    paramGrid = GridParametersOneVsRest.getParamsGrid(estimator.asInstanceOf[OneVsRest], classifier, labelColumn, featureColumn, predictionColumn, bernoulliOption)
    this
  }

  override def defineValidatorModel(): CrossValidationOneVsRestTask = {
    crossValidator = new CrossValidator().setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setEstimator(estimator).setNumFolds(numFolds)
    this
  }
}