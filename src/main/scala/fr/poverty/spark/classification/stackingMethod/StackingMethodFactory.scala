package fr.poverty.spark.classification.stackingMethod

import org.apache.spark.sql.{DataFrame, SparkSession}

trait StackingMethodFactory {

  def run(spark: SparkSession): StackingMethodFactory

  def defineValidationModel(data: DataFrame): StackingMethodFactory

  def transform(data: DataFrame): DataFrame

  def saveModel(path: String): StackingMethodFactory

  def loadModel(path: String): StackingMethodFactory
}
