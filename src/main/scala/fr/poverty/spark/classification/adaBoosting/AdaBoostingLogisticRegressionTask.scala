package fr.poverty.spark.classification.adaBoosting

import fr.poverty.spark.classification.gridParameters.GridParametersLogisticRegression
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.sql.{DataFrame, SparkSession}
import fr.poverty.spark.classification.validation.ValidationObject

class AdaBoostingLogisticRegressionTask(override val idColumn: String, override val labelColumn: String, override val featureColumn: String,
                                        override val predictionColumn: String, override val weightColumn: String,
                                        override val numberOfWeakClassifier: Int,
                                        override val pathSave: String,
                                        override val validationMethod: String, override val ratio: Double)
  extends AdaBoostingTask(idColumn, labelColumn, featureColumn, predictionColumn, weightColumn, numberOfWeakClassifier, pathSave, validationMethod, ratio) with AdaBoostingFactory{

  private var model: LogisticRegression = _
  private var weakClassifierList: List[LogisticRegressionModel] = List()
  private var optimalWeakClassifierList: List[LogisticRegressionModel] = List()
  private var validationError: Double = Double.NaN
  private var training: DataFrame = _
  private var validation: DataFrame = _

  def run(spark: SparkSession, data: DataFrame): AdaBoostingLogisticRegressionTask = {
    computeNumberOfObservation(data)
    computeNumberOfClass(data)
    computeInitialObservationWeight(data)
    loopWeakClassifier(spark, data)
    this
  }

  def trainValidationSplit(data: DataFrame): AdaBoostingLogisticRegressionTask = {
    val trainingValidation = ValidationObject.trainValidationSplit(data, ratio)
    training = trainingValidation(0)
    validation = trainingValidation(1)
    this
  }

  def gridParameters(): Array[(Double, Double)] = {
    val regParam = GridParametersLogisticRegression.getRegParam
    val elasticParam = GridParametersLogisticRegression.getElasticNetParam
    for(reg <- regParam; elastic <- elasticParam) yield (reg, elastic)
  }

  def loopGridParameters(spark: SparkSession): Unit = {
    gridParameters().foreach(param => {
      defineModel(param(0), param(1))
      weakClassifierList = List()
      weightWeakClassifierList = List()
      loopWeakClassifier(spark, training)

    })
  }

  def loopWeakClassifier(spark: SparkSession, data: DataFrame): AdaBoostingLogisticRegressionTask = {
    var weightError: Double = 1.0
    var index: Int = 1
    var weightData: DataFrame = addInitialWeightColumn(data)
    while(weightError > 1e-6 && index <= numberOfWeakClassifier) {
      val modelFitted = model.fit(weightData)
      weakClassifierList = weakClassifierList ++ List(modelFitted)
      weightData = modelFitted.transform(weightData)
      weightError = computeWeightError(weightData)
      val weightWeakClassifier = computeWeightWeakClassifier(weightData, weightError)
      weightWeakClassifierList = weightWeakClassifierList ++ List(weightWeakClassifier)
      weightData = updateWeightObservation(spark, weightData, weightWeakClassifier)
      index += 1
    }
    this
  }

  def defineModel(regParam: Double, elasticNetParam: Double): AdaBoostingLogisticRegressionTask = {
    model = new LogisticRegression()
      .setLabelCol(labelColumn)
      .setFeaturesCol(featureColumn)
      .setPredictionCol(predictionColumn)
      .setWeightCol(weightColumn)
      .setRegParam(regParam)
      .setElasticNetParam(elasticNetParam)
    this
  }

  def getModel: LogisticRegression = model

  def getWeakClassifierList: List[LogisticRegressionModel] = weakClassifierList
}
