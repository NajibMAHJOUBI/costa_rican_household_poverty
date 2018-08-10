package fr.poverty.spark.classification.adaBoosting

import org.apache.spark.ml.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.sql.{DataFrame, SparkSession}

class AdaBoostNaiveBayesTask(override val idColumn: String, override val labelColumn: String, override val featureColumn: String,
                                     override val predictionColumn: String, override val weightColumn: String,
                                     override val numberOfWeakClassifier: Int,
                                     override val pathSave: String)
  extends AdaBoostingTask(idColumn, labelColumn, featureColumn, predictionColumn, weightColumn, numberOfWeakClassifier, pathSave) with AdaBoostingFactory{

  private var model: NaiveBayes = _
  private var weightWeakClassifierList: List[Double] = List()
  private var weakClassifierList: List[NaiveBayesModel] = List()

  def run(spark: SparkSession, data: DataFrame): AdaBoostNaiveBayesTask = {
    computeNumberOfObservation(data)
    computeNumberOfClass(data)
    computeInitialObservationWeight(data)
    defineModel()
    loopWeakClassifier(spark, data)
    this
  }

  def loopWeakClassifier(spark: SparkSession, data: DataFrame): AdaBoostNaiveBayesTask = {
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

  def defineModel(): AdaBoostNaiveBayesTask = {
    model = new NaiveBayes()
      .setLabelCol(labelColumn)
      .setFeaturesCol(featureColumn)
      .setPredictionCol(predictionColumn)
      .setWeightCol(weightColumn)
    this
  }

  def getModel: NaiveBayes = model

}
