package fr.poverty.spark.classification.crossValidation

import fr.poverty.spark.classification.gridParameters.GridParametersOneVsRest
import fr.poverty.spark.classification.task.{CrossValidationModelFactory, OneVsRestTask}
import org.apache.spark.ml.classification.{OneVsRest, OneVsRestModel}
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.sql.DataFrame


class CrossValidationOneVsRestTask(override val labelColumn: String,
                                   override val featureColumn: String,
                                   override val predictionColumn: String,
                                   override val numFolds: Integer,
                                   override val pathSave: String,
                                   val classifier: String)
  extends CrossValidationTask(labelColumn, featureColumn, predictionColumn, numFolds, pathSave)
    with CrossValidationModelFactory {

  var estimator: OneVsRest = _

  override def run(data: DataFrame): CrossValidationOneVsRestTask = {
    defineEstimator()
    defineGridParameters()
    defineEvaluator()
    defineCrossValidatorModel()
    fit(data)
    this
  }

  override def defineEstimator(): CrossValidationOneVsRestTask = {
    estimator = new OneVsRestTask(labelColumn, featureColumn, predictionColumn, classifier).defineModel.getModel
    this
  }

  override def defineGridParameters(): CrossValidationOneVsRestTask = {
    paramGrid = GridParametersOneVsRest.getParamsGrid(estimator, classifier, labelColumn, featureColumn, predictionColumn)
    this
  }

  override def defineCrossValidatorModel(): CrossValidationOneVsRestTask = {
    crossValidator = new CrossValidator()
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setEstimator(estimator)
      .setNumFolds(numFolds)
    this
  }

  def getEstimator: OneVsRest = {
    estimator
  }

  def getBestModel: OneVsRestModel = {
    crossValidatorModel.bestModel.asInstanceOf[OneVsRestModel]
  }
}