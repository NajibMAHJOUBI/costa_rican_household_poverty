package fr.poverty.spark.classification.crossValidation

import fr.poverty.spark.classification.gridParameters.GridParametersNaiveBayes
import fr.poverty.spark.classification.task.{CrossValidationModelFactory, NaiveBayesTask}
import org.apache.spark.ml.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.DataFrame


class CrossValidationNaiveBayesTask(override val labelColumn: String,
                                    override val featureColumn: String,
                                    override val predictionColumn: String,
                                    override val numFolds: Integer,
                                    override val pathSave: String,
                                    val bernoulliOption: Boolean) extends CrossValidationTask(labelColumn, featureColumn, predictionColumn, numFolds, pathSave) with CrossValidationModelFactory {

  var estimator: NaiveBayes = _

  override def run(data: DataFrame): CrossValidationNaiveBayesTask = {
    defineEstimator()
    defineGridParameters()
    defineEvaluator()
    defineCrossValidatorModel()
    fit(data)
    saveModel()
    this
  }

  override def defineEstimator(): CrossValidationNaiveBayesTask = {
    estimator = new NaiveBayesTask(labelColumn=labelColumn,
                                   featureColumn=featureColumn,
                                   predictionColumn=predictionColumn).defineModel.getModel
    this
  }

  override def defineGridParameters(): CrossValidationNaiveBayesTask = {
    paramGrid = new ParamGridBuilder()
      .addGrid(estimator.modelType, GridParametersNaiveBayes.getModelType(bernoulliOption))
      .build()
    this
  }

  override def defineCrossValidatorModel(): CrossValidationNaiveBayesTask = {
    crossValidator = new CrossValidator()
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setEstimator(estimator)
      .setNumFolds(numFolds)
    this
  }

  def getEstimator: NaiveBayes = estimator

  def getBestModel: NaiveBayesModel = crossValidatorModel.bestModel.asInstanceOf[NaiveBayesModel]

}