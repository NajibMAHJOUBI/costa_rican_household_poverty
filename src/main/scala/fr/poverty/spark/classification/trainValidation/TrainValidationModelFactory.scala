package fr.poverty.spark.classification.trainValidation

import org.apache.spark.sql.DataFrame

trait TrainValidationModelFactory {

  def run(data: DataFrame): TrainValidationModelFactory

  def defineEstimator(): TrainValidationModelFactory

  def defineGridParameters(): TrainValidationModelFactory

  def defineTrainValidatorModel(): TrainValidationModelFactory

}
