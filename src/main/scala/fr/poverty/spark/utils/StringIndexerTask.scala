package fr.poverty.spark.utils

import org.apache.spark.ml.feature.{StringIndexer, StringIndexerModel}
import org.apache.spark.sql.DataFrame

class StringIndexerTask(val inputColumn: String, val outputColumn: String, val path: String) {

  var model: StringIndexer = _
  var modelFit: StringIndexerModel = _

  def run(data: DataFrame): DataFrame = {
    defineModel()
    fit(data)
    transform(data)
  }

  def defineModel(): StringIndexerTask= {
    model = new StringIndexer()
      .setInputCol(inputColumn)
      .setOutputCol(outputColumn)
    this
  }

  def fit(data: DataFrame): StringIndexerTask = {
    modelFit = model.fit(data)
    this
  }

  def transform(data: DataFrame): DataFrame = {
    modelFit.transform(data)
  }

  def saveModel(): StringIndexerTask = {
    modelFit.write.overwrite().save(s"$path/modelStringIndexer")
    this
  }

  def loadModel(path: String): StringIndexerModel = {
    StringIndexerModel.load(path)
  }

  def getLabels: Array[String] = modelFit.labels

}
