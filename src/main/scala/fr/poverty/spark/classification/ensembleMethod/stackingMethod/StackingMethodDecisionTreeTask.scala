package fr.poverty.spark.classification.ensembleMethod.stackingMethod

import fr.poverty.spark.classification.validation.crossValidation.CrossValidationDecisionTreeTask
import fr.poverty.spark.classification.validation.trainValidation.TrainValidationDecisionTreeTask
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.sql.{DataFrame, SparkSession}


class StackingMethodDecisionTreeTask(override val idColumn: String,
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

  override def run(spark: SparkSession): StackingMethodDecisionTreeTask = {
    predictionLabelFeatures = createLabelFeatures(spark, "prediction")
    submissionLabelFeatures = createLabelFeatures(spark, "submission")
    defineValidationModel(predictionLabelFeatures)
    transform()
    savePrediction()
    saveSubmission()
    this
  }

  override def defineValidationModel(data: DataFrame): StackingMethodDecisionTreeTask = {
    if (validationMethod == "crossValidation") {
      val cv = new CrossValidationDecisionTreeTask(labelColumn, featureColumn, predictionColumn = "prediction", metricName, pathSave, ratio.toInt)
      cv.run(data)
      model = cv.getBestModel
    } else if (validationMethod == "trainValidation") {
      val tv = new TrainValidationDecisionTreeTask(labelColumn, featureColumn, "prediction", metricName, pathSave, ratio.toDouble)
      tv.run(data)
      model = tv.getBestModel
    }
    this
  }

  override def saveModel(path: String): StackingMethodDecisionTreeTask = {
    model.asInstanceOf[DecisionTreeClassificationModel].write.overwrite().save(path)
    this
  }

  override def loadModel(path: String): StackingMethodDecisionTreeTask = {
    model = DecisionTreeClassificationModel.load(path)
    this
  }

}
