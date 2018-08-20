package fr.poverty.spark.classification.task

import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.sql.DataFrame

class RandomForestTask(override val labelColumn: String, override val featureColumn: String, override val predictionColumn: String)
  extends ClassificationModelTask(labelColumn, featureColumn, predictionColumn)
    with ClassificationModelFactory {

  var model: RandomForestClassifier = _
  var modelFit: RandomForestClassificationModel = _
  var transform: DataFrame = _

  override def defineModel: RandomForestTask= {
    model = new RandomForestClassifier().setFeaturesCol(featureColumn).setLabelCol(labelColumn).setPredictionCol(predictionColumn)
    this
  }

  override def fit(data: DataFrame): RandomForestTask = {
    modelFit = getModel.fit(data)
    this
  }

  def getModel: RandomForestClassifier = model

  override def transform(data: DataFrame): RandomForestTask = {
    prediction = modelFit.transform(data)
    this
  }

  override def loadModel(path: String): RandomForestTask = {
    modelFit = RandomForestClassificationModel.load(path)
    this
  }

  override def saveModel(path: String): RandomForestTask = {
    model.write.overwrite().save(path)
    this
  }

  def getModelFit: RandomForestClassificationModel = modelFit

}
