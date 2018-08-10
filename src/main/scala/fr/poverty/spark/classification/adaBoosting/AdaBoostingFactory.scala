package fr.poverty.spark.classification.adaBoosting

import org.apache.spark.sql.{DataFrame, SparkSession}

trait AdaBoostingFactory {

  def run(spark: SparkSession, data: DataFrame): AdaBoostingFactory

  def loopWeakClassifier(spark: SparkSession, data: DataFrame): AdaBoostingFactory

  def defineModel(): AdaBoostingFactory

}
