package fr.poverty.spark.classification.task

import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.sql.DataFrame

class DecisionTreeTask(override val labelColumn: String, override val featureColumn: String, override val predictionColumn: String) extends ClassificationModelTask(labelColumn, featureColumn, predictionColumn) with ClassificationModelFactory {

  var model: DecisionTreeClassifier = _
  var modelFit: DecisionTreeClassificationModel = _

  override def defineModel: DecisionTreeTask= {
    model = new DecisionTreeClassifier()
      .setFeaturesCol(featureColumn)
      .setLabelCol(labelColumn)
      .setPredictionCol(predictionColumn)
    this
  }

  override def fit(data: DataFrame): DecisionTreeTask = {
    modelFit = getModel.fit(data)
    this
  }

  override def transform(data: DataFrame): DecisionTreeTask = {
    prediction = modelFit.transform(data)
    this
  }

  override def loadModel(path: String): DecisionTreeTask = {
    modelFit = DecisionTreeClassificationModel.load(path)
    this
  }

  override def saveModel(path: String): DecisionTreeTask = {
    model.write.overwrite().save(path)
    this
  }

  def getModel: DecisionTreeClassifier = model

  def getModelFit: DecisionTreeClassificationModel = modelFit

}
