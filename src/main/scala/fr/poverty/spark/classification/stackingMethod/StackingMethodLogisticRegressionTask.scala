package fr.poverty.spark.classification.stackingMethod

import fr.poverty.spark.classification.crossValidation.CrossValidationLogisticRegressionTask
import fr.poverty.spark.classification.trainValidation.TrainValidationLogisticRegressionTask
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.sql.{DataFrame, SparkSession}

class StackingMethodLogisticRegressionTask(override val pathPrediction: List[String], override val formatPrediction: String,
                                           override val pathTrain: String, override val formatTrain: String,
                                           override val pathSave: String,
                                           override val validationMethod: String,
                                           override val ratio: Double,
                                           override  val idColumn: String,
                                           override val labelColumn: String,
                                           override val predictionColumn: String)
  extends StackingMethodTask(pathPrediction, formatPrediction, pathTrain, formatTrain, pathSave, validationMethod, ratio, idColumn,
    labelColumn, predictionColumn)
    with StackingMethodFactory {

  val featureColumn: String = "features"
  var model: LogisticRegressionModel = _

  override def run(spark: SparkSession): StackingMethodLogisticRegressionTask = {
    labelFeatures = new StackingMethodTask(pathPrediction, formatPrediction, pathTrain,
      formatTrain, pathSave, validationMethod, ratio, idColumn, labelColumn, predictionColumn).createLabelFeatures(spark)
    defineValidationModel(labelFeatures)
    this
  }

  override def defineValidationModel(data: DataFrame): StackingMethodLogisticRegressionTask = {
    if (validationMethod == "crossValidation") {
      val cv = new CrossValidationLogisticRegressionTask(labelColumn,
        featureColumn, "prediction", numFolds = ratio.toInt,
        pathSave = "")
      cv.run(data)
      model = cv.getBestModel
    } else if (validationMethod == "trainValidation") {
      val tv = new TrainValidationLogisticRegressionTask(labelColumn, featureColumn,
        "prediction", trainRatio=ratio.toDouble, "")
      tv.run(data)
      model = tv.getBestModel
    }
    this
  }


  override def transform(data: DataFrame): DataFrame = {
    model.transform(data)
  }

  override def saveModel(path: String): StackingMethodLogisticRegressionTask = {
    model.write.overwrite().save(path)
    this
  }

  def loadModel(path: String): StackingMethodLogisticRegressionTask = {
    model = LogisticRegressionModel.load(path)
    this
  }
}
