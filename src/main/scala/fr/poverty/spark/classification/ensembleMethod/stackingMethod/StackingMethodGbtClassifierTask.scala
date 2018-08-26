package fr.poverty.spark.classification.ensembleMethod.stackingMethod

import fr.poverty.spark.classification.validation.crossValidation.CrossValidationGbtClassifierTask
import fr.poverty.spark.classification.validation.trainValidation.TrainValidationGbtClassifierTask
import org.apache.spark.ml.classification.GBTClassificationModel
import org.apache.spark.sql.{DataFrame, SparkSession}


class StackingMethodGbtClassifierTask(override val idColumn: String,
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

  override def run(spark: SparkSession): StackingMethodGbtClassifierTask = {
    predictionLabelFeatures = createLabelFeatures(spark, "prediction")
    submissionLabelFeatures = createLabelFeatures(spark, "submission")
    defineValidationModel(predictionLabelFeatures)
    transform()
    savePrediction()
    saveSubmission()
    this
  }

  override def defineValidationModel(data: DataFrame): StackingMethodGbtClassifierTask = {
    if (validationMethod == "crossValidation") {
      val cv = new CrossValidationGbtClassifierTask(labelColumn, featureColumn, "prediction", metricName, pathSave, ratio.toInt)
      cv.run(data)
      model = cv.getBestModel
    } else if (validationMethod == "trainValidation") {
      val tv = new TrainValidationGbtClassifierTask(labelColumn, featureColumn, "prediction", metricName, pathSave, ratio.toDouble)
      tv.run(data)
      model = tv.getBestModel
    }
    this
  }

  override def saveModel(path: String): StackingMethodGbtClassifierTask = {
    model.asInstanceOf[GBTClassificationModel].write.overwrite().save(path)
    this
  }

  def loadModel(path: String): StackingMethodGbtClassifierTask = {
    model = GBTClassificationModel.load(path)
    this
  }


}
