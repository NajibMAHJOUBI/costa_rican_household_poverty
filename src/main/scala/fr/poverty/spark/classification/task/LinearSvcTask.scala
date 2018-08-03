package fr.poverty.spark.classification.task

import org.apache.spark.ml.classification.{LinearSVC, LinearSVCModel}
import org.apache.spark.sql.DataFrame

/**
  * Created by mahjoubi on 12/06/18.
  *
  * LinearSVC classifier
  *
  */
class LinearSvcTask(override val labelColumn: String, override val featureColumn: String, override val predictionColumn: String) extends ClassificationModelTask(labelColumn, featureColumn, predictionColumn) with ClassificationModelFactory {

  var model: LinearSVC = _
  var modelFit: LinearSVCModel = _

  override def defineModel: LinearSvcTask= {
    model = new LinearSVC()
      .setFeaturesCol(featureColumn)
      .setLabelCol(labelColumn)
      .setPredictionCol(predictionColumn)
    this
  }

  override def fit(data: DataFrame): LinearSvcTask = {
    modelFit = getModel.fit(data)
    this
  }

  def getModel: LinearSVC = model

  override def transform(data: DataFrame): LinearSvcTask = {
    prediction = modelFit.transform(data)
    this
  }

  override def loadModel(path: String): LinearSvcTask = {
    modelFit = LinearSVCModel.load(path)
    this
  }

  override def saveModel(path: String): LinearSvcTask = {
    model.write.overwrite().save(path)
    this
  }

  def getModelFit: LinearSVCModel = modelFit

}
