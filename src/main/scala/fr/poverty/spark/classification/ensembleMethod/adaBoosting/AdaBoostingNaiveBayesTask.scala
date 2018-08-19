package fr.poverty.spark.classification.ensembleMethod.adaBoosting

import fr.poverty.spark.classification.evaluation.EvaluationObject
import fr.poverty.spark.classification.gridParameters.GridParametersNaiveBayes
import fr.poverty.spark.classification.validation.ValidationObject
import org.apache.spark.ml.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.sql.{DataFrame, SparkSession}

class AdaBoostingNaiveBayesTask(override val idColumn: String, override val labelColumn: String, override val featureColumn: String,
                                override val predictionColumn: String, override val weightColumn: String,
                                override val numberOfWeakClassifier: Int,
                                override val pathSave: String,
                                override val validationMethod: String, override val ratio: Double, val bernoulliOption: Boolean)
  extends AdaBoostingTask(idColumn, labelColumn, featureColumn, predictionColumn, weightColumn, numberOfWeakClassifier, pathSave, validationMethod, ratio) with AdaBoostingFactory{

  private var model: NaiveBayes = _
  private var weakClassifierList: List[NaiveBayesModel] = List()
  private var optimalWeakClassifierList: List[NaiveBayesModel] = List()

  def run(spark: SparkSession, data: DataFrame): AdaBoostingNaiveBayesTask = {
    computeNumberOfClass(data)
    trainValidationSplit(data)
    loopGridParameters(spark)
    this
  }

  def trainValidationSplit(data: DataFrame): AdaBoostingNaiveBayesTask = {
    val trainingValidation = ValidationObject.trainValidationSplit(data, ratio)
    training = trainingValidation(0)
    validation = trainingValidation(1)
    this
  }

  def gridParameters(): Array[String] = {
    val modelTypeParam = GridParametersNaiveBayes.getModelType(bernoulliOption)
    for(modelType <- modelTypeParam) yield (modelType)
  }

  def loopGridParameters(spark: SparkSession): Unit = {
    var oldValidationError: Double = Double.NaN
    computeInitialObservationWeight(training)
    gridParameters().foreach(param => {
      defineModel(param)
      weakClassifierList = List()
      weightClassifierList = List()
      loopWeakClassifier(spark, training)
      val prediction = computePrediction(spark, validation, weakClassifierList, weightClassifierList)
      val newValidationError = EvaluationObject.defineMultiClassificationEvaluator(labelColumn, predictionColumn).evaluate(prediction)
      println(newValidationError)
      if(oldValidationError.isNaN || newValidationError < oldValidationError){
        oldValidationError = newValidationError
        optimalWeakClassifierList = weakClassifierList
        optimalWeightClassifierList = weightClassifierList
      }
    })
  }

  def loopWeakClassifier(spark: SparkSession, data: DataFrame): AdaBoostingNaiveBayesTask = {
    var weightError: Double = 1.0
    var index: Int = 1
    var weightData: DataFrame = addInitialWeightColumn(data)
    while(weightError > 1e-6 && index <= numberOfWeakClassifier) {
      val modelFitted = model.fit(weightData)
      weakClassifierList = weakClassifierList ++ List(modelFitted)
      weightData = modelFitted.transform(weightData)
      weightError = computeWeightError(weightData)
      val weightWeakClassifier = computeWeightWeakClassifier(weightData, weightError)
      weightClassifierList = weightClassifierList ++ List(weightWeakClassifier)
      weightData = updateWeightObservation(spark, weightData, weightWeakClassifier)
      index += 1
    }
    this
  }

  def defineModel(modelType: String): AdaBoostingNaiveBayesTask = {
    model = new NaiveBayes()
      .setLabelCol(labelColumn)
      .setFeaturesCol(featureColumn)
      .setPredictionCol(predictionColumn)
      .setWeightCol(weightColumn)
      .setModelType(modelType)
    this
  }

  def getModel: NaiveBayes = model

  def getWeakClassifierList: List[NaiveBayesModel] = optimalWeakClassifierList

}
