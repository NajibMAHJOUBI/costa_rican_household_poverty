package fr.poverty.spark.classification.task

import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.sql.DataFrame

/**
  * Created by mahjoubi on 12/06/18.
  */
class LogisticRegressionTask(override val labelColumn: String, override val featureColumn: String, override val predictionColumn: String) extends ClassificationModelTask(labelColumn, featureColumn, predictionColumn) with ClassificationModelFactory {

  var model: LogisticRegression = _
  var modelFit: LogisticRegressionModel = _

  def getModelFit: LogisticRegressionModel = {
    modelFit
  }

  override def defineModel: LogisticRegressionTask= {
    model = new LogisticRegression()
      .setFeaturesCol(featureColumn)
      .setLabelCol(labelColumn)
      .setPredictionCol(predictionColumn)
    this
  }

  override def fit(data: DataFrame): LogisticRegressionTask = {
    modelFit = getModel.fit(data)
    this
  }

  def getModel: LogisticRegression = {
    model
  }

  override def transform(data: DataFrame): LogisticRegressionTask = {
    prediction = modelFit.transform(data)
    this
  }

  override def saveModel(path: String): LogisticRegressionTask = {
    model.write.overwrite().save(path)
    this
  }

  override def loadModel(path: String): LogisticRegressionTask = {
    modelFit = LogisticRegressionModel.load(path)
    this
  }

}
