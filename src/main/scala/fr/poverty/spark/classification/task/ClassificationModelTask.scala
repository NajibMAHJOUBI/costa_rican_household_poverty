package fr.poverty.spark.classification.task

import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.DataFrame

class ClassificationModelTask(val labelColumn: String, val featureColumn: String, val predictionColumn: String) {

  var estimator: Estimator[_] = _
  var model: Model[_] = _

  def fit(data: DataFrame): ClassificationModelTask = {
    model = estimator.fit(data).asInstanceOf[Model[_]]
    this
  }

  def transform(data: DataFrame): DataFrame = {
    model.transform(data)
  }

  def getEstimator: Estimator[_] = estimator

  def getModel: Model[_] = model

}
