package fr.poverty.spark.classification.trainValidation

import fr.poverty.spark.classification.task.{DecisionTreeTask, OneVsRestTask}
import fr.poverty.spark.classification.gridParameters.GridParametersOneVsRest
import org.apache.spark.ml.classification.{OneVsRest, OneVsRestModel}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.DataFrame


class TrainValidationOneVsRestTask(override val labelColumn: String,
                                   override val featureColumn: String,
                                   override val predictionColumn: String,
                                   override val trainRatio: Double,
                                   override val pathSave: String,
                                   val classifier: String,
                                   val bernoulliOption: Boolean = false)
  extends TrainValidationTask(labelColumn, featureColumn, predictionColumn, trainRatio, pathSave)
    with TrainValidationModelFactory {

  var estimator: OneVsRest = _

  override def run(data: DataFrame): TrainValidationOneVsRestTask = {
    defineEstimator()
    defineGridParameters()
    defineEvaluator()
    defineTrainValidatorModel()
    fit(data)
    saveModel()
    this
  }

  override def defineEstimator(): TrainValidationOneVsRestTask = {
    estimator = new OneVsRestTask(labelColumn, featureColumn, predictionColumn, classifier).defineModel.getModel
    this
  }

  override def defineGridParameters(): TrainValidationOneVsRestTask = {
    paramGrid = GridParametersOneVsRest.getParamsGrid(estimator, classifier, labelColumn, featureColumn, predictionColumn, bernoulliOption)
    this
  }

  override def defineTrainValidatorModel(): TrainValidationOneVsRestTask = {
    trainValidator = new TrainValidationSplit()
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setEstimator(estimator)
      .setTrainRatio(trainRatio)
    this
  }

  def getEstimator: OneVsRest = {
    estimator
  }

  def getBestModel: OneVsRestModel = {
    trainValidatorModel.bestModel.asInstanceOf[OneVsRestModel]
  }
}