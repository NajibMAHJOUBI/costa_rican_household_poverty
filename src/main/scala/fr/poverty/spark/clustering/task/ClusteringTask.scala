package fr.poverty.spark.clustering.task

import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.DataFrame


class ClusteringTask(val featuresColumn: String, val predictionColumn: String) {

  var estimator: Estimator[_] = _
  var model: Model[_] = _

  def fit(data: DataFrame): ClusteringTask = {
    model = estimator.fit(data).asInstanceOf[Model[_]]
    this
  }

  def transform(data: DataFrame): DataFrame = {
    model.transform(data)
  }

}
