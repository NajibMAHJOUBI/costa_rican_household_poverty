package fr.poverty.spark.classification.stackingMethod

import fr.poverty.spark.classification.validation.crossValidation.CrossValidationOneVsRestTask
import fr.poverty.spark.classification.validation.trainValidation.TrainValidationOneVsRestTask
import org.apache.spark.ml.classification.OneVsRestModel
import org.apache.spark.sql.{DataFrame, SparkSession}


class StackingMethodOneVsRestTask(override val idColumn: String, override val labelColumn: String, override val predictionColumn: String,
                                  override val pathPrediction: List[String], override val formatPrediction: String,
                                  override val pathTrain: String, override val formatTrain: String,
                                  override val pathStringIndexer: String, override val pathSave: String,
                                  override val validationMethod: String, override val ratio: Double, val classifier: String, val bernoulliOption: Boolean)
  extends StackingMethodTask(idColumn, labelColumn, predictionColumn, pathPrediction, formatPrediction, pathTrain, formatTrain, pathStringIndexer, pathSave, validationMethod, ratio)
    with StackingMethodFactory {

  val featureColumn: String = "features"
  var model: OneVsRestModel = _

  override def run(spark: SparkSession): StackingMethodOneVsRestTask = {
    labelFeatures = new StackingMethodTask(idColumn, labelColumn, predictionColumn, pathPrediction, formatPrediction, pathTrain, formatTrain, pathStringIndexer, pathSave, validationMethod, ratio).createLabelFeatures(spark)
    defineValidationModel(labelFeatures)
    this
  }

  override def defineValidationModel(data: DataFrame): StackingMethodOneVsRestTask = {
    if (validationMethod == "crossValidation") {
      val cv = new CrossValidationOneVsRestTask(labelColumn = labelColumn,
        featureColumn = featureColumn, predictionColumn = "prediction", numFolds = ratio.toInt,
        pathSave = "", classifier = classifier, bernoulliOption = bernoulliOption)
      cv.run(data)
      model = cv.getBestModel
    } else if (validationMethod == "trainValidation") {
      val tv = new TrainValidationOneVsRestTask(labelColumn, featureColumn,
        "prediction", "", trainRatio=ratio.toDouble, classifier, bernoulliOption)
      tv.run(data)
      model = tv.getBestModel
    }
    this
  }

  override def transform(data: DataFrame): DataFrame = model.transform(data)

  override def saveModel(path: String): StackingMethodOneVsRestTask = {
    model.write.overwrite().save(path)
    this
  }

  override def loadModel(path: String): StackingMethodOneVsRestTask = {
    model = OneVsRestModel.load(path)
    this
  }
}
