package fr.poverty.spark.kaggle

import fr.poverty.spark.classification.adaBoosting.AdaBoostingLogisticRegressionTask
import fr.poverty.spark.utils._
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.SparkSession

import scala.io.Source

object KaggleAdaBoostingExample {

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
    val weightColumn = "weight"
    val numberOfWeakClassifierList = List(5) // , 10, 15
    val sourcePath = "src/main/resources"
    val savePath = "submission/adaBoosting"
    val models = List("logisticRegression", "naiveBayes")
    val validationMethod: String = "trainValidation"
    val ratio: Double = 0.70

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
    val labelFeaturesSubmission = new DefineLabelFeaturesTask(idColumn, "", sourcePath).run(spark, testFilled)

    val stringIndexer = new StringIndexerTask(targetColumn, labelColumn, "")
    val labelFeaturesIndexed = stringIndexer.run(labelFeatures)

    val indexToString = new IndexToStringTask(predictionColumn, targetColumn, stringIndexer.getLabels)

    numberOfWeakClassifierList.foreach(numberOfWeakClassifier => {
      println(s"number of weak classifier: $numberOfWeakClassifier")
      models.foreach(model => {
        if(model == "logisticRegression"){
          println(s"Model: $model")
          val logisticRegression = new AdaBoostingLogisticRegressionTask(idColumn, labelColumn, featureColumn,
            predictionColumn, weightColumn, numberOfWeakClassifier, s"$savePath/weakClassifier_$numberOfWeakClassifier/$model",
            validationMethod, ratio)
          logisticRegression.run(spark, labelFeaturesIndexed)
          val prediction = logisticRegression.computePrediction(spark, labelFeaturesIndexed, logisticRegression.getWeakClassifierList)
//          val submission = logisticRegression.computeSubmission(spark, labelFeaturesSubmission, logisticRegression.getWeakClassifierList)
          logisticRegression.savePrediction(indexToString.run(prediction))
//          logisticRegression.saveSubmission(indexToString.run(submission))
        }
      })
    })



  }
}
