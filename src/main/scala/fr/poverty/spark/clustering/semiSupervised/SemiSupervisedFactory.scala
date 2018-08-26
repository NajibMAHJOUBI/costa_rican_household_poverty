package fr.poverty.spark.clustering.semiSupervised

import org.apache.spark.sql.{DataFrame, SparkSession}

trait SemiSupervisedFactory {

  def run(spark: SparkSession, train: DataFrame, test: DataFrame): SemiSupervisedFactory

  def defineModel(data: DataFrame): SemiSupervisedFactory

}
