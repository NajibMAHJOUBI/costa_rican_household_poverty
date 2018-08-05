package fr.poverty.spark.classification.validation

import org.apache.spark.ml.evaluation.{Evaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.IntegerType

class ValidationTask(val labelColumn: String,
                     val featureColumn: String,
                     val predictionColumn: String,
                     val pathSave: String) {

  var evaluator: MulticlassClassificationEvaluator = _
  var paramGrid: Array[ParamMap] = _
  var prediction: DataFrame = _

  def savePrediction(data: DataFrame): Unit = {
    data.write.mode("overwrite").parquet(s"$pathSave/prediction")
  }

  def saveSubmission(data: DataFrame, idColumn: String, predictionColumn: String): Unit = {
    data
      .select(col(idColumn), col(predictionColumn).cast(IntegerType))
      .repartition(1)
      .write
      .option("header", "true")
      .option("delimiter", ",")
      .mode("overwrite")
      .csv(s"$pathSave/submission")
  }

  def defineEvaluator(): ValidationTask = {
    evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol(labelColumn)
      .setPredictionCol(predictionColumn)
      .setMetricName("accuracy")
    this
  }

  def getLabelColumn: String = labelColumn

  def getFeatureColumn: String = featureColumn

  def getPredictionColumn: String = predictionColumn

  def getGridParameters: Array[ParamMap] = paramGrid

  def getEvaluator: Evaluator = evaluator

  def getPrediction: DataFrame = prediction

  }
