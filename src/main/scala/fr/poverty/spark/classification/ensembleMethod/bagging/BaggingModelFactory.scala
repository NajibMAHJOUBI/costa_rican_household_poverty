package fr.poverty.spark.classification.ensembleMethod.bagging

import org.apache.spark.ml.Model
import org.apache.spark.sql.DataFrame

trait BaggingModelFactory {

  def run(data: DataFrame): Unit

  def loopDataSampling(): Unit

  def getModels: List[Model[_]]

}
