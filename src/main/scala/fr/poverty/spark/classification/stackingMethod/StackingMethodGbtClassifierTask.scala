package fr.poverty.spark.classification.stackingMethod

import fr.poverty.spark.classification.crossValidation.CrossValidationGbtClassifierTask
import fr.poverty.spark.classification.trainValidation.TrainValidationGbtClassifierTask
import org.apache.spark.ml.classification.GBTClassificationModel
import org.apache.spark.sql.{DataFrame, SparkSession}


class StackingMethodGbtClassifierTask(override val pathPrediction: List[String], override val formatPrediction: String,
                                      override val pathTrain: String, override val formatTrain: String,
                                      override val pathSave: String,
                                      override val validationMethod: String,
                                      override val ratio: Double,
                                      override  val idColumn: String,
                                      override val labelColumn: String,
                                      override val predictionColumn: String,
                                      val bernoulliOption: Boolean)
  extends StackingMethodTask(pathPrediction, formatPrediction, pathTrain, formatTrain, pathSave, validationMethod, ratio, idColumn,
    labelColumn, predictionColumn)
    with StackingMethodFactory {

  val featureColumn: String = "features"
  var model: GBTClassificationModel = _

  override def run(spark: SparkSession): StackingMethodGbtClassifierTask = {
    labelFeatures = new StackingMethodTask(pathPrediction, formatPrediction, pathTrain,
      formatTrain, pathSave, validationMethod, ratio, idColumn, labelColumn, predictionColumn).createLabelFeatures(spark)
    defineValidationModel(labelFeatures)
    this
  }

  override def defineValidationModel(data: DataFrame): StackingMethodGbtClassifierTask = {
    if (validationMethod == "crossValidation") {
      val cv = new CrossValidationGbtClassifierTask(labelColumn = labelColumn,
        featureColumn = featureColumn, predictionColumn = "prediction", numFolds = ratio.toInt, pathSave = "")
      cv.run(data)
      model = cv.getBestModel
    } else if (validationMethod == "trainValidation") {
      val tv = new TrainValidationGbtClassifierTask(labelColumn, featureColumn,
        "prediction", trainRatio=ratio.toDouble, "")
      tv.run(data)
      model = tv.getBestModel
    }
    this
  }

  override def transform(data: DataFrame): DataFrame = {
    model.transform(data)
  }

  override def saveModel(path: String): StackingMethodGbtClassifierTask = {
    model.write.overwrite().save(path)
    this
  }

  def loadModel(path: String): StackingMethodGbtClassifierTask = {
    model = GBTClassificationModel.load(path)
    this
  }
}
