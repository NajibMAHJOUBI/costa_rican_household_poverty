package fr.poverty.spark.classification.validation

import org.apache.spark.sql.DataFrame

trait ValidationModelFactory {

  def run(data: DataFrame): ValidationModelFactory

  def defineEstimator(): ValidationModelFactory

  def defineGridParameters(): ValidationModelFactory

  def defineValidatorModel(): ValidationModelFactory

}
