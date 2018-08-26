package fr.poverty.spark.classification.validation.crossValidation

import fr.poverty.spark.classification.gridParameters.GridParametersNaiveBayes
import fr.poverty.spark.classification.task.NaiveBayesTask
import fr.poverty.spark.classification.validation.ValidationModelFactory
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.sql.DataFrame


class CrossValidationNaiveBayesTask(override val labelColumn: String,
                                    override val featureColumn: String,
                                    override val predictionColumn: String,
                                    override val metricName: String,
                                    override val pathSave: String,
                                    override val numFolds: Integer,
                                    val bernoulliOption: Boolean)
  extends CrossValidationTask(labelColumn, featureColumn, predictionColumn, metricName, pathSave, numFolds)
    with ValidationModelFactory {

  override def run(data: DataFrame): CrossValidationNaiveBayesTask = {
    defineEstimator()
    defineGridParameters()
    defineEvaluator()
    defineValidatorModel()
    fit(data)
    this
  }

  override def defineEstimator(): CrossValidationNaiveBayesTask = {
    estimator = new NaiveBayesTask(labelColumn=labelColumn, featureColumn=featureColumn, predictionColumn=predictionColumn).defineEstimator.getEstimator
    this
  }

  override def defineGridParameters(): CrossValidationNaiveBayesTask = {
    paramGrid = GridParametersNaiveBayes.getParamsGrid(estimator.asInstanceOf[NaiveBayes], bernoulliOption)
    this
  }

  override def defineValidatorModel(): CrossValidationNaiveBayesTask = {
    crossValidator = new CrossValidator().setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setEstimator(estimator).setNumFolds(numFolds)
    this
  }

}