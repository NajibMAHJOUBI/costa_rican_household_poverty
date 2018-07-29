package fr.poverty.spark.kaggle

import fr.poverty.spark.classification.trainValidation._
import fr.poverty.spark.utils.{DefineLabelFeaturesTask, LoadDataSetTask, ReplacementNoneValuesTask, StringIndexerTask}
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.SparkSession

import scala.io.Source


object KaggleTrainValidationExample {

  def main(arguments: Array[String]): Unit = {
    val spark = SparkSession.builder.master("local").appName("Kaggle Submission Example - Classification methods").getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)

    // --> Initialization
    val idColumn = "Id"
    val targetColumn = "Target"
    val labelColumn = "label"
    val featureColumn = "features"
    val predictionColumn = "prediction"
    val trainRatio = 0.75
    val models = Array("naiveBayes") //Array("decisionTree", "randomForest", "logisticRegression", "oneVsRest")
    val sourcePath = "src/main/resources"
    val savePath = "submission/trainValidation"

    // --> features name
    val nullFeatures = Source.fromFile(s"$sourcePath/nullFeaturesNames").getLines.toList.head.split(",")
    val yesNoFeatures = Source.fromFile(s"$sourcePath/yesNoFeaturesNames").getLines.toList.head.split(",")

    // --> Train and Test data set
    val train = new LoadDataSetTask(sourcePath = "data", format = "csv").run(spark, "train")
    val test = new LoadDataSetTask(sourcePath = "data", format = "csv").run(spark, "test")

    val replacementNoneValues = new ReplacementNoneValuesTask(targetColumn, nullFeatures, yesNoFeatures).run(spark, train, test)
    val trainFilled = replacementNoneValues.getTrain
    val testFilled = replacementNoneValues.getTest

    val labelFeatures = new DefineLabelFeaturesTask(idColumn, targetColumn, sourcePath).run(spark, trainFilled)
    val labelFeaturesSubmission = new DefineLabelFeaturesTask(idColumn, targetColumn, sourcePath).run(spark, testFilled)

    val stringIndexer = new StringIndexerTask(targetColumn, labelColumn, savePath)
    val labelFeaturesIndexed = stringIndexer.run(labelFeatures)

    models.foreach(model =>{
      if (model == "decisionTree") {
        val decisionTree = new TrainValidationDecisionTreeTask(labelColumn, featureColumn,predictionColumn, trainRatio,
          s"$savePath/$model")
        decisionTree.run(labelFeaturesIndexed)
        decisionTree.transform(labelFeaturesIndexed)
        decisionTree.savePrediction()
        decisionTree.transform(labelFeaturesSubmission)
        decisionTree.saveSubmission()
      }
      else if (model == "randomForest") {
        val randomForest = new TrainValidationRandomForestTask(labelColumn, featureColumn,predictionColumn, trainRatio,
          s"$savePath/$model")
        randomForest.run(labelFeaturesIndexed)
        randomForest.transform(labelFeaturesIndexed)
        randomForest.savePrediction()
        randomForest.transform(labelFeaturesSubmission)
        randomForest.saveSubmission()
      }
      else if (model == "logisticRegression") {
        val logisticRegression = new TrainValidationLogisticRegressionTask(labelColumn, featureColumn, predictionColumn,
          trainRatio, s"$savePath/$model")
        logisticRegression.run(labelFeaturesIndexed)
        logisticRegression.transform(labelFeaturesIndexed)
        logisticRegression.savePrediction()
        logisticRegression.transform(labelFeaturesSubmission)
        logisticRegression.saveSubmission()
      }
      else if (model == "oneVsRest") {
        val oneVsRest = new TrainValidationOneVsRestTask(labelColumn, featureColumn, predictionColumn,
          trainRatio, s"$savePath/$model", "logisticRegression")
        oneVsRest.run(labelFeaturesIndexed)
        oneVsRest.transform(labelFeaturesIndexed)
        oneVsRest.savePrediction()
        oneVsRest.transform(labelFeaturesSubmission)
        oneVsRest.saveSubmission()
      }
      else if (model == "naiveBayes") {
        val naiveBayes = new TrainValidationNaiveBayesTask(labelColumn, featureColumn, predictionColumn,
          trainRatio, s"$savePath/$model", false)
        naiveBayes.run(labelFeaturesIndexed)
        naiveBayes.transform(labelFeaturesIndexed)
        naiveBayes.savePrediction()
        naiveBayes.transform(labelFeaturesSubmission)
        naiveBayes.saveSubmission()}
    }
    )
  }
}
