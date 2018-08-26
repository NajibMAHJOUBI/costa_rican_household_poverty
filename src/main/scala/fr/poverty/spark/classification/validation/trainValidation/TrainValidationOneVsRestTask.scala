package fr.poverty.spark.classification.validation.trainValidation

import fr.poverty.spark.classification.gridParameters.GridParametersOneVsRest
import fr.poverty.spark.classification.task.OneVsRestTask
import fr.poverty.spark.classification.validation.ValidationModelFactory
import org.apache.spark.ml.classification.OneVsRest
import org.apache.spark.ml.tuning.TrainValidationSplit
import org.apache.spark.sql.DataFrame


class TrainValidationOneVsRestTask(override val labelColumn: String,
                                   override val featureColumn: String,
                                   override val predictionColumn: String,
                                   override val metricName: String,
                                   override val pathSave: String,
                                   override val trainRatio: Double,
                                   val classifier: String,
                                   val bernoulliOption: Boolean = false)
  extends TrainValidationTask(labelColumn, featureColumn, predictionColumn, metricName, pathSave, trainRatio)
    with ValidationModelFactory {

  override def run(data: DataFrame): TrainValidationOneVsRestTask = {
    defineEstimator()
    defineGridParameters()
    defineEvaluator()
    defineValidatorModel()
    fit(data)
    this
  }

  override def defineEstimator(): TrainValidationOneVsRestTask = {
    estimator = new OneVsRestTask(labelColumn, featureColumn, predictionColumn, classifier).defineEstimator.getEstimator
    this
  }

  override def defineGridParameters(): TrainValidationOneVsRestTask = {
    paramGrid = GridParametersOneVsRest.getParamsGrid(estimator.asInstanceOf[OneVsRest], classifier, labelColumn, featureColumn, predictionColumn, bernoulliOption)
    this
  }

  override def defineValidatorModel(): TrainValidationOneVsRestTask = {
    trainValidator = new TrainValidationSplit()
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setEstimator(estimator)
      .setTrainRatio(trainRatio)
    this
  }

}