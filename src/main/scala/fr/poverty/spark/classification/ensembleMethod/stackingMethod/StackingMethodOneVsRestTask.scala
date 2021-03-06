package fr.poverty.spark.classification.ensembleMethod.stackingMethod

import fr.poverty.spark.classification.validation.crossValidation.CrossValidationOneVsRestTask
import fr.poverty.spark.classification.validation.trainValidation.TrainValidationOneVsRestTask
import org.apache.spark.ml.classification.OneVsRestModel
import org.apache.spark.sql.{DataFrame, SparkSession}


class StackingMethodOneVsRestTask(override val idColumn: String,
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
                                  val classifier: String,
                                  val bernoulliOption: Boolean)
  extends StackingMethodTask(idColumn, labelColumn, predictionColumn, pathPrediction, mapFormat, pathTrain, formatTrain, pathStringIndexer, pathSave, validationMethod, ratio, metricName)
    with StackingMethodFactory {

  override def run(spark: SparkSession): StackingMethodOneVsRestTask = {
    predictionLabelFeatures = createLabelFeatures(spark, "prediction")
    submissionLabelFeatures = createLabelFeatures(spark, "submission")
    defineValidationModel(predictionLabelFeatures)
    transform()
    savePrediction()
    saveSubmission()
    this
  }

  override def defineValidationModel(data: DataFrame): StackingMethodOneVsRestTask = {
    if (validationMethod == "crossValidation") {
      val cv = new CrossValidationOneVsRestTask(labelColumn, featureColumn, "prediction", metricName, pathSave, ratio.toInt, classifier,
        bernoulliOption)
      cv.run(data)
      model = cv.getBestModel
    } else if (validationMethod == "trainValidation") {
      val tv = new TrainValidationOneVsRestTask(labelColumn, featureColumn, "prediction", metricName, pathSave, ratio.toDouble, classifier,
        bernoulliOption)
      tv.run(data)
      model = tv.getBestModel
    }
    this
  }

  override def saveModel(path: String): StackingMethodOneVsRestTask = {
    model.asInstanceOf[OneVsRestModel].write.overwrite().save(path)
    this
  }

  override def loadModel(path: String): StackingMethodOneVsRestTask = {
    model = OneVsRestModel.load(path)
    this
  }

}
