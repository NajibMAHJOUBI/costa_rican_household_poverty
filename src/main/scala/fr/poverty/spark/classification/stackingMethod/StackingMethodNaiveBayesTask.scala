package fr.poverty.spark.classification.stackingMethod

import fr.poverty.spark.classification.validation.crossValidation.CrossValidationNaiveBayesTask
import fr.poverty.spark.classification.validation.trainValidation.TrainValidationNaiveBayesTask
import org.apache.spark.ml.classification.NaiveBayesModel
import org.apache.spark.sql.{DataFrame, SparkSession}


class StackingMethodNaiveBayesTask(override val pathPrediction: List[String], override val formatPrediction: String,
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
  var model: NaiveBayesModel = _

  override def run(spark: SparkSession): StackingMethodNaiveBayesTask = {
    labelFeatures = new StackingMethodTask(pathPrediction, formatPrediction, pathTrain,
      formatTrain, pathSave, validationMethod, ratio, idColumn, labelColumn, predictionColumn).createLabelFeatures(spark)
    defineValidationModel(labelFeatures)
    this
  }

  override def defineValidationModel(data: DataFrame): StackingMethodNaiveBayesTask = {
    if (validationMethod == "crossValidation") {
      val cv = new CrossValidationNaiveBayesTask(labelColumn = labelColumn,
        featureColumn = featureColumn, predictionColumn = "prediction", numFolds = ratio.toInt, pathSave = "",
        bernoulliOption = bernoulliOption)
      cv.run(data)
      model = cv.getBestModel
    } else if (validationMethod == "trainValidation") {
      val tv = new TrainValidationNaiveBayesTask(labelColumn, featureColumn,
        "prediction", "", trainRatio=ratio.toDouble, bernoulliOption)
      tv.run(data)
      model = tv.getBestModel
    }
    this
  }

  override def transform(data: DataFrame): DataFrame = model.transform(data)

  override def saveModel(path: String): StackingMethodNaiveBayesTask = {
    model.write.overwrite().save(path)
    this
  }

  def loadModel(path: String): StackingMethodNaiveBayesTask = {
    model = NaiveBayesModel.load(path)
    this
  }
}
