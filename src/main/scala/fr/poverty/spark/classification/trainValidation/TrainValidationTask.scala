package fr.poverty.spark.classification.trainValidation

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{TrainValidationSplit, TrainValidationSplitModel}
import org.apache.spark.sql.DataFrame

class TrainValidationTask(val labelColumn: String, val featureColumn: String, val predictionColumn: String, val trainRatio: Double, val pathSave: String) {

  var prediction: DataFrame = _
  var paramGrid: Array[ParamMap] = _
  var trainValidator: TrainValidationSplit = _
  var trainValidatorModel: TrainValidationSplitModel = _
  var evaluator: MulticlassClassificationEvaluator = _


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
    prediction.select("Id", "Target").write.option("header", "true").mode("overwrite").csv(s"$pathSave/submission")
  }

}
