package fr.poverty.spark.utils

import org.apache.spark.sql.DataFrame

import scala.io.Source

class DefineLabelFeaturesTask(val labelColumn: String) {

  private var featureNames: Array[String] = _

  def run(data: DataFrame): Unit = {
    featureNames = readFeatureNames()

  }

  def readFeatureNames(): Array[String] = {
    Source.fromFile("src/main/resources/featuresNames").getLines.toList(0).split(",")
  }



}
