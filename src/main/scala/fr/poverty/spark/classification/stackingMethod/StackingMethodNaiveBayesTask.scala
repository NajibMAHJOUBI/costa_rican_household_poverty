package fr.poverty.spark.classification.stackingMethod

import fr.poverty.spark.classification.validation.crossValidation.CrossValidationNaiveBayesTask
import fr.poverty.spark.classification.validation.trainValidation.TrainValidationNaiveBayesTask
import org.apache.spark.ml.classification.NaiveBayesModel
import org.apache.spark.sql.{DataFrame, SparkSession}


class StackingMethodNaiveBayesTask(override val idColumn: String, override val labelColumn: String, override val predictionColumn: String,
                                   override val pathPrediction: List[String], override val mapFormat: Map[String, String],
                                   override val pathTrain: String, override val formatTrain: String,
                                   override val pathStringIndexer: String, override val pathSave: String,
                                   override val validationMethod: String, override val ratio: Double,
                                   val bernoulliOption: Boolean)
  extends StackingMethodTask(idColumn, labelColumn, predictionColumn, pathPrediction, mapFormat, pathTrain, formatTrain, pathStringIndexer, pathSave, validationMethod, ratio)
    with StackingMethodFactory {

  val featureColumn: String = "features"
  var model: NaiveBayesModel = _

  override def run(spark: SparkSession): StackingMethodNaiveBayesTask = {
    predictionLabelFeatures = createLabelFeatures(spark, "prediction")
    submissionLabelFeatures = createLabelFeatures(spark, "submission")
    defineValidationModel(predictionLabelFeatures)
    transform()
    savePrediction()
    saveSubmission()
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

  override def saveModel(path: String): StackingMethodNaiveBayesTask = {
    model.write.overwrite().save(path)
    this
  }

  def loadModel(path: String): StackingMethodNaiveBayesTask = {
    model = NaiveBayesModel.load(path)
    this
  }

  override def transform(): StackingMethodNaiveBayesTask = {
    transformPrediction = model.transform(predictionLabelFeatures).drop(labelColumn)
    transformSubmission = model.transform(submissionLabelFeatures)
    this
  }

}
