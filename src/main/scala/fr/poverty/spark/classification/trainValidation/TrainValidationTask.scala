package fr.poverty.spark.classification.trainValidation

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{TrainValidationSplit, TrainValidationSplitModel}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.IntegerType

class TrainValidationTask(val labelColumn: String, val featureColumn: String, val predictionColumn: String, val trainRatio: Double, val pathSave: String) {

  var prediction: DataFrame = _
  var paramGrid: Array[ParamMap] = _
  var evaluator: MulticlassClassificationEvaluator = _
  var trainValidator: TrainValidationSplit = _
  var trainValidatorModel: TrainValidationSplitModel = _

  def fit(data: DataFrame): TrainValidationTask = {
    trainValidatorModel = trainValidator.fit(data)
    this
  }

  def transform(data: DataFrame): TrainValidationTask = {
    prediction = trainValidatorModel.transform(data)
    this
  }

  def defineEvaluator(): TrainValidationTask = {
    evaluator = new MulticlassClassificationEvaluator().setLabelCol(labelColumn).setPredictionCol(predictionColumn).setMetricName("accuracy")
    this
  }

  def saveModel(): Unit = {
    trainValidatorModel.write.overwrite().save(s"$pathSave/model")
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

  def getEvaluator: MulticlassClassificationEvaluator = {
    evaluator
  }

  def getParamGrid: Array[ParamMap] = {
    paramGrid
  }

  def getTrainValidator: TrainValidationSplit = {
    trainValidator
  }

  def getTrainValidatorModel: TrainValidationSplitModel = {
    trainValidatorModel
  }

}
