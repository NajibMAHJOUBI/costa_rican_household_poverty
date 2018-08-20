package fr.poverty.spark.classification.evaluation


import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

object EvaluationObject {

  def defineMultiClassificationEvaluator(labelColumn: String, predictionColumn: String, metricName: String): MulticlassClassificationEvaluator = {
    new MulticlassClassificationEvaluator()
      .setLabelCol(labelColumn)
      .setPredictionCol(predictionColumn)
      .setMetricName(metricName)
  }


}
