package fr.poverty.spark.sampleMinorityClass

import org.apache.spark.sql.{DataFrame, SparkSession}

trait SampleMinorityClassFactory {

  def run(spark: SparkSession, data: DataFrame): DataFrame

  def resampleClass(spark: SparkSession, data: DataFrame, classIdentifier: Int, objectiveSize: Long): DataFrame

  def resampleDataSet(spark: SparkSession, data: DataFrame, mapClassCount: Map[Int, Long], objectiveSize: Long): DataFrame

}
