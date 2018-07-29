package fr.poverty.spark.classification.crossValidation

import fr.poverty.spark.classification.gridParameters.GridParametersRandomForest
import fr.poverty.spark.classification.task.{CrossValidationModelFactory, RandomForestTask}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.DataFrame


class CrossValidationRandomForestTask(override val labelColumn: String,
                                      override val featureColumn: String,
                                      override val predictionColumn: String,
                                      override val pathSave: String) extends CrossValidationTask(labelColumn, featureColumn, predictionColumn, pathSave) with CrossValidationModelFactory {

  var estimator: RandomForestClassifier = _

  override def run(data: DataFrame): CrossValidationRandomForestTask = {
    defineEstimator()
    defineGridParameters()
    defineEvaluator()
    defineCrossValidatorModel()
    fit(data)
    this
  }

  override def defineEstimator(): CrossValidationRandomForestTask = {
    estimator = new RandomForestTask(labelColumn=labelColumn,
                                     featureColumn=featureColumn,
                                     predictionColumn=predictionColumn).defineModel.getModel
    this
  }

  override def defineGridParameters(): CrossValidationRandomForestTask = {
      paramGrid = new ParamGridBuilder()
        .addGrid(estimator.maxDepth, GridParametersRandomForest.getMaxDepth)
        .addGrid(estimator.maxBins, GridParametersRandomForest.getMaxBins)
        .build()
    this
  }

  override def defineCrossValidatorModel(): CrossValidationRandomForestTask = {
    crossValidator = new CrossValidator()
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setEstimator(estimator)
    this
  }

  def getEstimator: RandomForestClassifier = estimator

  def getBestModel: RandomForestClassificationModel = crossValidatorModel.bestModel.asInstanceOf[RandomForestClassificationModel]

}