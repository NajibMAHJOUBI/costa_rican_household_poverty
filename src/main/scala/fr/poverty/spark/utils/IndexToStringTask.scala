package fr.poverty.spark.utils

import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.sql.DataFrame


class IndexToStringTask(val inputColumn: String, val outputColumn: String, val labels: Array[String]){

  var model: IndexToString = _

  def run(data: DataFrame): DataFrame = {
    defineModel()
    transform(data)
  }

  def defineModel(): IndexToStringTask = {
    model = new IndexToString()
        .setInputCol(inputColumn)
        .setOutputCol(outputColumn)
        .setLabels(labels)
    this
  }

  def transform(data: DataFrame): DataFrame = {
    model.transform(data)
  }

}
