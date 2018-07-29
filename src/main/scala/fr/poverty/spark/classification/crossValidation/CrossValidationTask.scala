package fr.poverty.spark.classification.crossValidation

import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, Evaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.IntegerType

class CrossValidationTask(val labelColumn: String,
                          val featureColumn: String,
                          val predictionColumn: String,
                          val pathSave: String) {

  var evaluator: MulticlassClassificationEvaluator = _
  var paramGrid: Array[ParamMap] = _
  var crossValidator: CrossValidator = _
  var crossValidatorModel: CrossValidatorModel = _
  var prediction: DataFrame = _

  def fit(data: DataFrame): CrossValidationTask = {
    crossValidatorModel = crossValidator.fit(data)
    this
  }

  def transform(data: DataFrame): CrossValidationTask = {
    prediction = crossValidatorModel.transform(data)
    this
  }

  def saveModel(): Unit = {
    crossValidatorModel.write.overwrite().save(s"$pathSave/model")
  }

  def savePrediction(): Unit = {
    prediction.write.parquet(s"$pathSave/prediction")
  }

  def saveSubmission(): Unit = {
    prediction
      .select(col("Id"), col("prediction").cast(IntegerType).alias("Target"))
      .repartition(1)
      .write
      .option("header", "true")
      .option("delimiter", ",")
      .mode("overwrite")
      .csv(s"$pathSave/submission")
  }

  def defineEvaluator(): CrossValidationTask = {
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

  def getCrossValidator: CrossValidator = crossValidator

  def getCrossValidatorModel: CrossValidatorModel = crossValidatorModel

  def getPrediction: DataFrame = prediction

  def setGridParameters(grid: Array[ParamMap]): CrossValidationTask = {
    paramGrid = grid
    this
  }
  }
