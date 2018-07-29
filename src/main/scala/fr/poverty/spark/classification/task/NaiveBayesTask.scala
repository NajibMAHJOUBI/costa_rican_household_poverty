package fr.poverty.spark.classification.task

import org.apache.spark.ml.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.sql.DataFrame

class NaiveBayesTask(override val labelColumn: String, override val featureColumn: String, override val predictionColumn: String) extends ClassificationModelTask(labelColumn, featureColumn, predictionColumn) with ClassificationModelFactory {

  var model: NaiveBayes = _
  var modelFit: NaiveBayesModel = _
  var transform: DataFrame = _

  override def defineModel: NaiveBayesTask= {
    model = new NaiveBayes()
      .setFeaturesCol(featureColumn)
      .setLabelCol(labelColumn)
      .setPredictionCol(predictionColumn)
    this
  }

  override def fit(data: DataFrame): NaiveBayesTask = {
    modelFit = getModel.fit(data)
    this
  }

  def getModel: NaiveBayes = {
    model
  }

  override def transform(data: DataFrame): NaiveBayesTask = {
    prediction = modelFit.transform(data)
    this
  }

  override def loadModel(path: String): NaiveBayesTask = {
    modelFit = NaiveBayesModel.load(path)
    this
  }

  override def saveModel(path: String): NaiveBayesTask = {
    model.write.overwrite().save(path)
    this
  }

  def getModelFit: NaiveBayesModel = {
    modelFit
  }

}
