package fr.poverty.spark.classification.task

import org.apache.spark.sql.DataFrame

trait CrossValidationModelFactory {

  def run(data: DataFrame): CrossValidationModelFactory

  def defineEstimator(): CrossValidationModelFactory

  def defineGridParameters(): CrossValidationModelFactory

  def defineCrossValidatorModel(): CrossValidationModelFactory

  def getLabelColumn: String

  def getFeatureColumn: String

  def getPredictionColumn: String

}
