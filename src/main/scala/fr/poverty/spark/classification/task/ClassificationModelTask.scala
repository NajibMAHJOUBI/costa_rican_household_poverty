package fr.poverty.spark.classification.task

import org.apache.spark.sql.DataFrame

class ClassificationModelTask(val labelColumn: String, val featureColumn: String, val predictionColumn: String) {

  var prediction: DataFrame = _

  def getPrediction: DataFrame = {
    prediction
  }

}
