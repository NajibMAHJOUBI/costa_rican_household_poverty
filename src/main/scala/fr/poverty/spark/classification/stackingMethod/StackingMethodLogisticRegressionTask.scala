package fr.poverty.spark.classification.stackingMethod

import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.sql.{DataFrame, SparkSession}

class StackingMethodLogisticRegressionTask(override val classificationMethod: String,
                                           override val pathPrediction: List[String], override val formatPrediction: String,
                                           override val pathTrain: String, override val formatTrain: String,
                                           override val pathSave: String,
                                           override val validationMethod: String,
                                           override val ratio: Double,
                                           override  val idColumn: String,
                                           override val labelColumn: String,
                                           override val predictionColumn: String)
  extends StackingMethodTask(classificationMethod, pathPrediction, formatPrediction, pathTrain, formatTrain, pathSave, validationMethod, ratio, idColumn,
    labelColumn, predictionColumn)
    with StackingMethodFactory {

  val featureColumn: String = "features"
  var model: LogisticRegressionModel = _

  override def run(spark: SparkSession): StackingMethodLogisticRegressionTask = {
    this
  }

  override def computeModel(data: DataFrame, label: String): StackingMethodLogisticRegressionTask = {
    this
  }

  override def saveModel(column: String): StackingMethodLogisticRegressionTask = {
    this
  }

  override def computePrediction(data: DataFrame): StackingMethodLogisticRegressionTask = {
    this
  }

  def loadModel(path: String): StackingMethodLogisticRegressionTask = {
    this
  }
}
