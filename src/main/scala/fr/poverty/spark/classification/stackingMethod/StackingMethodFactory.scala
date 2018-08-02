package fr.poverty.spark.classification.stackingMethod

import org.apache.spark.sql.{DataFrame, SparkSession}

trait StackingMethodFactory {

  def run(spark: SparkSession): StackingMethodFactory

  def defineModel(): StackingMethodFactory

  def fit(data: DataFrame): StackingMethodFactory

  def transform(data: DataFrame): StackingMethodFactory

  def saveModel(path: String): StackingMethodFactory

  def loadModel(path: String): StackingMethodFactory
}
