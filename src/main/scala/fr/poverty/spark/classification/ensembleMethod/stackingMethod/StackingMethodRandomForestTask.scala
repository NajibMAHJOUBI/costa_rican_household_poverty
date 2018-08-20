package fr.poverty.spark.classification.ensembleMethod.stackingMethod


import fr.poverty.spark.classification.validation.crossValidation.CrossValidationRandomForestTask
import fr.poverty.spark.classification.validation.trainValidation.TrainValidationRandomForestTask
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.sql.{DataFrame, SparkSession}


class StackingMethodRandomForestTask(override val idColumn: String,
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
                                     override val metricName: String)
  extends StackingMethodTask(idColumn, labelColumn, predictionColumn, pathPrediction, mapFormat, pathTrain, formatTrain, pathStringIndexer, pathSave, validationMethod, ratio, metricName)
    with StackingMethodFactory {

  val featureColumn: String = "features"
  var model: RandomForestClassificationModel = _

  override def run(spark: SparkSession): StackingMethodRandomForestTask = {
    predictionLabelFeatures = createLabelFeatures(spark, "prediction")
    submissionLabelFeatures = createLabelFeatures(spark, "submission")
    defineValidationModel(predictionLabelFeatures)
    transform()
    savePrediction()
    saveSubmission()
    this
  }

  override def defineValidationModel(data: DataFrame): StackingMethodRandomForestTask = {
    if (validationMethod == "crossValidation") {
      val cv = new CrossValidationRandomForestTask(labelColumn, featureColumn, predictionColumn, metricName, pathSave, ratio.toInt)
      cv.run(data)
      model = cv.getBestModel
    } else if (validationMethod == "trainValidation") {
      val tv = new TrainValidationRandomForestTask(labelColumn, featureColumn, predictionColumn, metricName, pathSave, ratio.toDouble)
      tv.run(data)
      model = tv.getBestModel
    }
    this
  }

  override def saveModel(path: String): StackingMethodRandomForestTask = {
    model.write.overwrite().save(path)
    this
  }

  override def loadModel(path: String): StackingMethodRandomForestTask = {
    model = RandomForestClassificationModel.load(path)
    this
  }

  override def transform(): StackingMethodRandomForestTask = {
    transformPrediction = model.transform(predictionLabelFeatures).drop(labelColumn)
    transformSubmission = model.transform(submissionLabelFeatures)
    this
  }
}
