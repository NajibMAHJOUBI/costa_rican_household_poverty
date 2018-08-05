package fr.poverty.spark.classification.stackingMethod


import fr.poverty.spark.classification.validation.crossValidation.CrossValidationRandomForestTask
import fr.poverty.spark.classification.validation.trainValidation.TrainValidationRandomForestTask
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.sql.{DataFrame, SparkSession}


class StackingMethodRandomForestTask(override val pathPrediction: List[String], override val formatPrediction: String,
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
  var model: RandomForestClassificationModel = _

  override def run(spark: SparkSession): StackingMethodRandomForestTask = {
    labelFeatures = new StackingMethodTask(pathPrediction, formatPrediction, pathTrain,
      formatTrain, pathSave, validationMethod, ratio, idColumn, labelColumn, predictionColumn).createLabelFeatures(spark)
    defineValidationModel(labelFeatures)
    this
  }

  override def defineValidationModel(data: DataFrame): StackingMethodRandomForestTask = {
    if (validationMethod == "crossValidation") {
      val cv = new CrossValidationRandomForestTask(labelColumn = labelColumn,
        featureColumn = featureColumn, predictionColumn = "prediction", numFolds = ratio.toInt,
        pathSave = "")
      cv.run(data)
      model = cv.getBestModel
    } else if (validationMethod == "trainValidation") {
      val tv = new TrainValidationRandomForestTask(labelColumn, featureColumn,
        "prediction", "", trainRatio=ratio.toDouble)
      tv.run(data)
      model = tv.getBestModel
    }
    this
  }

  override def transform(data: DataFrame): DataFrame = model.transform(data)

  override def saveModel(path: String): StackingMethodRandomForestTask = {
    model.write.overwrite().save(path)
    this
  }

  override def loadModel(path: String): StackingMethodRandomForestTask = {
    model = RandomForestClassificationModel.load(path)
    this
  }
}
