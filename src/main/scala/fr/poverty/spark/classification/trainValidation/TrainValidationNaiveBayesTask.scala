package fr.poverty.spark.classification.trainValidation

import fr.poverty.spark.classification.gridParameters.GridParametersNaiveBayes
import fr.poverty.spark.classification.task.NaiveBayesTask
import org.apache.spark.ml.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.ml.tuning.TrainValidationSplit
import org.apache.spark.sql.DataFrame


class TrainValidationNaiveBayesTask(override val labelColumn: String, override val featureColumn: String, override val predictionColumn: String, override val trainRatio: Double, override val pathSave: String, val bernoulliOption: Boolean) extends TrainValidationTask(labelColumn, featureColumn, predictionColumn, trainRatio, pathSave) with TrainValidationModelFactory {

  var estimator: NaiveBayes = _

  override def run(data: DataFrame): TrainValidationNaiveBayesTask = {
    defineEstimator()
    defineGridParameters()
    defineEvaluator()
    defineTrainValidatorModel()
    fit(data)
    this
  }

  override def defineEstimator(): TrainValidationNaiveBayesTask = {
    estimator = new NaiveBayesTask(labelColumn = labelColumn, featureColumn = featureColumn, predictionColumn = predictionColumn).defineModel.getModel
    this
  }

  override def defineGridParameters(): TrainValidationNaiveBayesTask = {
    paramGrid = GridParametersNaiveBayes.getParamsGrid(estimator, bernoulliOption)
    this
  }

  override def defineTrainValidatorModel(): TrainValidationNaiveBayesTask = {
    trainValidator = new TrainValidationSplit().setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setEstimator(estimator).setTrainRatio(trainRatio)
    this
  }

  def getEstimator: NaiveBayes = {
    estimator
  }

  def getBestModel: NaiveBayesModel = {
    trainValidatorModel.bestModel.asInstanceOf[NaiveBayesModel]
  }

}