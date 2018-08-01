package fr.poverty.spark.classification.stackingMethod

import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.sql.{DataFrame, SparkSession}


class StackingMethodDecisionTreeTask(override val classificationMethods: Array[String],
                                     override val pathTrain: String,
                                     override val pathPrediction: String,
                                     override val pathSave: String,
                                     override val validationMethod: String,
                                     override val ratio: Double,
                                     override val idColumn: String,
                                     override val labelColumn: String,
                                     override val predictionColumn: String,
                                     val numFolds: Int,
                                     val trainRation: Double)
  extends StackingMethodTask(classificationMethods, pathTrain, pathPrediction, pathSave, validationMethod, ratio, idColumn,
    labelColumn, predictionColumn)
    with StackingMethodFactory {

  val featureColumn: String = "features"
  var model: DecisionTreeClassificationModel = _

  override def run(spark: SparkSession): StackingMethodDecisionTreeTask = {
    this
  }

  override def computeModel(data: DataFrame, label: String): StackingMethodDecisionTreeTask = {
    this
  }

  override def saveModel(column: String): StackingMethodDecisionTreeTask = {
    this
  }

  override def computePrediction(data: DataFrame): StackingMethodDecisionTreeTask = {
    this
  }

  def loadModel(path: String): StackingMethodDecisionTreeTask = {
    this
  }
}
