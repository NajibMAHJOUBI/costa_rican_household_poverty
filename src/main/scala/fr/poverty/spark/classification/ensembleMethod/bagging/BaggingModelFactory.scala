package fr.poverty.spark.classification.ensembleMethod.bagging

import org.apache.spark.sql.DataFrame

trait BaggingModelFactory {

  def run(data: DataFrame): Unit

  def loopDataSampling(): Unit

}
