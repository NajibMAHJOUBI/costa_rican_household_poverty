package fr.poverty.spark.classification.bagging

import org.apache.spark.sql.{DataFrame, SparkSession}
import scala.util.Random

class BaggingTask(val idColumn: String, val labelColumn: String, val featureColumn: String,
                  val predictionColumn: String, val pathSave: String,
                  val numberOfSampling: Int, val samplingFraction: Double,
                  val validationMethod: String, val ratio: Double) {

  private var sampleSubsetsList: List[DataFrame] = List()

  def run(spark: SparkSession, data: DataFrame): BaggingTask = {
    this
  }

  def defineSampleSubset(data: DataFrame): BaggingTask = {
    (1 to numberOfSampling).foreach(index => {
      sampleSubsetsList = sampleSubsetsList ++ List(data.sample(true, samplingFraction))
    })
    this
  }

  def getSampleSubset: List[DataFrame] = sampleSubsetsList

}
