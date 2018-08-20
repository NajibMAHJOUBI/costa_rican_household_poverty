package fr.poverty.spark.classification.task

import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.sql.DataFrame

/**
  * Created by mahjoubi on 12/06/18.
  */
class GbtClassifierTask(override val labelColumn: String, override val featureColumn: String, override val predictionColumn: String)
  extends ClassificationModelTask(labelColumn, featureColumn, predictionColumn)
    with ClassificationModelFactory {

  var model: GBTClassifier = _
  var modelFit: GBTClassificationModel = _

  def getModelFit: GBTClassificationModel = modelFit

  override def defineModel: GbtClassifierTask= {
    model = new GBTClassifier()
      .setFeaturesCol(featureColumn)
      .setLabelCol(labelColumn)
      .setPredictionCol(predictionColumn)
    this
  }

  override def fit(data: DataFrame): GbtClassifierTask = {
    modelFit = getModel.fit(data)
    this
  }

  def getModel: GBTClassifier = model

  override def transform(data: DataFrame): GbtClassifierTask = {
    prediction = modelFit.transform(data)
    this
  }

  override def saveModel(path: String): GbtClassifierTask = {
    model.write.overwrite().save(path)
    this
  }

  override def loadModel(path: String): GbtClassifierTask = {
    modelFit = GBTClassificationModel.load(path)
    this
  }

}
