package fr.poverty.spark.kaggle

import fr.poverty.spark.classification.trainValidation._
import fr.poverty.spark.utils.{DefineLabelFeaturesTask, LoadDataSetTask, ReplacementNoneValuesTask}
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.SparkSession

import scala.io.Source


object KaggleTrainValidationExample {

  def main(arguments: Array[String]): Unit = {
    val spark = SparkSession.builder.master("local").appName("Kaggle Submission Example - Classification methods").getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)

    // --> features name
    val nullFeatures = Source.fromFile("src/main/resources/nullFeaturesNames").getLines.toList(0).split(",")
    val yesNoFeatures = Source.fromFile("src/main/resources/yesNoFeaturesNames").getLines.toList(0).split(",")

    // --> Train and Test data set
    val train = new LoadDataSetTask(sourcePath = "data", format = "csv").run(spark, "train")
    val test = new LoadDataSetTask(sourcePath = "data", format = "csv").run(spark, "test")

    val replacementNoneValues = new ReplacementNoneValuesTask("target", nullFeatures, yesNoFeatures).run(spark, train, test)
    val trainFilled = replacementNoneValues.getTrain
    val testFilled = replacementNoneValues.getTest

    val labelFeatures = new DefineLabelFeaturesTask("Id", "Target", "src/main/resources").run(spark, trainFilled)
    val labelFeaturesSubmission = new DefineLabelFeaturesTask("Id", "", "src/main/resources").run(spark, testFilled)

    val models = Array("naiveBayes") //Array("decisionTree", "randomForest", "logisticRegression", "oneVsRest")
    val path = "submission/trainValidation"
    models.foreach(model =>{
      if (model == "decisionTree") {
        val decisionTree = new TrainValidationDecisionTreeTask("Target",
          "features",
          "prediction",
          0.75,
          s"$path/$model")
        decisionTree.run(labelFeatures)
        decisionTree.transform(labelFeatures)
        decisionTree.savePrediction()
        decisionTree.transform(labelFeaturesSubmission)
        decisionTree.saveSubmission()
      } else if (model == "randomForest") {
        val randomForest = new TrainValidationRandomForestTask("Target",
          "features",
          "prediction",
          0.75,
          s"$path/$model")
        randomForest.run(labelFeatures)
        randomForest.transform(labelFeatures)
        randomForest.savePrediction()
        randomForest.transform(labelFeaturesSubmission)
        randomForest.saveSubmission()
      } else if (model == "logisticRegression") {
        val logisticRegression = new TrainValidationLogisticRegressionTask("Target",
          "features",
          "prediction",
          0.75,
          s"$path/$model")
        logisticRegression.run(labelFeatures)
        logisticRegression.transform(labelFeatures)
        logisticRegression.savePrediction()
        logisticRegression.transform(labelFeaturesSubmission)
        logisticRegression.saveSubmission()
      } else if (model == "oneVsRest") {
        val oneVsRest = new TrainValidationOneVsRestTask("Target",
          "features",
          "prediction",
          0.75,
          s"$path/$model", "logisticRegression")
        oneVsRest.run(labelFeatures)
        oneVsRest.transform(labelFeatures)
        oneVsRest.savePrediction()
        oneVsRest.transform(labelFeaturesSubmission)
        oneVsRest.saveSubmission()
      } else if (model == "naiveBayes") {
        val naiveBayes = new TrainValidationNaiveBayesTask("Target",
          "features",
          "prediction",
          0.75,
          s"$path/$model")
        naiveBayes.run(labelFeatures)
        naiveBayes.transform(labelFeatures)
        naiveBayes.savePrediction()
        naiveBayes.transform(labelFeaturesSubmission)
        naiveBayes.saveSubmission()}

    }
    )
  }
}