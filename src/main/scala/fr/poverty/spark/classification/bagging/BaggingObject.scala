package fr.poverty.spark.classification.bagging

import org.apache.spark.sql.Row

object BaggingObject {

  def mergePredictions(p: Row, numberOfPrediction: Int): Double = {
    var predictions: List[Double] = List()
    (0 until numberOfPrediction).foreach(index => {
      predictions = predictions ++ List(p.getDouble(p.fieldIndex(s"prediction_$index")))
    })
    predictions.groupBy(i => i).mapValues(_.size).maxBy(_._2)._1
  }

}
