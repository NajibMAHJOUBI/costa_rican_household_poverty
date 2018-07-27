package fr.poverty.spark.classification.trainValidation

import fr.poverty.spark.classification.task.RandomForestTask
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.DataFrame


class TrainValidationRandomForestTask(override val labelColumn: String, override val featureColumn: String, override val predictionColumn: String, override val trainRatio: Double, override val pathSave: String) extends TrainValidationTask(labelColumn, featureColumn, predictionColumn, trainRatio, pathSave) with TrainValidationModelFactory {

  var estimator: RandomForestClassifier = _

  override def run(data: DataFrame): TrainValidationRandomForestTask = {
    defineEstimator()
    defineGridParameters()
    defineEvaluator()
    defineTrainValidatorModel()
    fit(data)
    saveModel()
    this
  }

  override def defineEstimator(): TrainValidationRandomForestTask = {
    estimator = new RandomForestTask(labelColumn = labelColumn, featureColumn = featureColumn, predictionColumn = predictionColumn).defineModel.getModel
    this
  }

  override def defineGridParameters(): TrainValidationRandomForestTask = {
    paramGrid = new ParamGridBuilder().addGrid(estimator.maxDepth, Array(4, 8, 16, 30)).addGrid(estimator.maxBins, Array(2, 4, 8, 16)).build()
    this
  }

  override def defineTrainValidatorModel(): TrainValidationRandomForestTask = {
    trainValidator = new TrainValidationSplit().setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setEstimator(estimator).setTrainRatio(trainRatio)
    this
  }

  def getEstimator: RandomForestClassifier = {
    estimator
  }

  def getBestModel: RandomForestClassificationModel = {
    trainValidatorModel.bestModel.asInstanceOf[RandomForestClassificationModel]
  }

}