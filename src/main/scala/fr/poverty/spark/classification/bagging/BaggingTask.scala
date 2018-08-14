package fr.poverty.spark.classification.bagging

import org.apache.spark.sql.{DataFrame, Row}

class BaggingTask(val idColumn: String, val labelColumn: String, val featureColumn: String,
                  val predictionColumn: String, val pathSave: String,
                  val numberOfSampling: Int, val samplingFraction: Double,
                  val validationMethod: String, val ratio: Double) {

  var sampleSubsetsList: List[DataFrame] = List()

  def defineSampleSubset(data: DataFrame): BaggingTask = {
    (1 to numberOfSampling).foreach(index => {
      sampleSubsetsList = sampleSubsetsList ++ List(data.sample(true, samplingFraction))
    })
    this
  }

  def mergePredictions(p: Row, numberOfPrediction: Int): Double = {
    var predictions: List[Double] = List()
    (0 until numberOfPrediction).foreach(index => {
      predictions = predictions ++ List(p.getDouble(p.fieldIndex(s"prediction_$index")))
    })
    predictions.groupBy(i => i).mapValues(_.size).maxBy(_._2)._1
  }

}
