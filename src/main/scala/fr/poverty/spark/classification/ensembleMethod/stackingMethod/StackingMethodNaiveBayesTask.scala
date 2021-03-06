package fr.poverty.spark.classification.ensembleMethod.stackingMethod

import fr.poverty.spark.classification.validation.crossValidation.CrossValidationNaiveBayesTask
import fr.poverty.spark.classification.validation.trainValidation.TrainValidationNaiveBayesTask
import org.apache.spark.ml.classification.NaiveBayesModel
import org.apache.spark.sql.{DataFrame, SparkSession}


class StackingMethodNaiveBayesTask(override val idColumn: String,
                                   override val labelColumn: String,
                                   override val predictionColumn: String,
                                   override val pathPrediction: List[String],
                                   override val mapFormat: Map[String, String],
                                   override val pathTrain: String,
                                   override val formatTrain: String,
                                   override val pathStringIndexer: String,
                                   override val pathSave: String,
                                   override val validationMethod: String,
                                   override val ratio: Double,
                                   override val metricName: String,
                                   val bernoulliOption: Boolean)
  extends StackingMethodTask(idColumn, labelColumn, predictionColumn, pathPrediction, mapFormat, pathTrain, formatTrain, pathStringIndexer, pathSave, validationMethod, ratio, metricName)
    with StackingMethodFactory {

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
      val cv = new CrossValidationNaiveBayesTask(labelColumn, featureColumn, "prediction", metricName, pathSave, ratio.toInt,
        bernoulliOption)
      cv.run(data)
      model = cv.getBestModel
    } else if (validationMethod == "trainValidation") {
      val tv = new TrainValidationNaiveBayesTask(labelColumn, featureColumn, "prediction", metricName, pathSave, trainRatio=ratio.toDouble,
        bernoulliOption)
      tv.run(data)
      model = tv.getBestModel
    }
    this
  }

  override def saveModel(path: String): StackingMethodNaiveBayesTask = {
    model.asInstanceOf[NaiveBayesModel].write.overwrite().save(path)
    this
  }

  def loadModel(path: String): StackingMethodNaiveBayesTask = {
    model = NaiveBayesModel.load(path)
    this
  }

}
