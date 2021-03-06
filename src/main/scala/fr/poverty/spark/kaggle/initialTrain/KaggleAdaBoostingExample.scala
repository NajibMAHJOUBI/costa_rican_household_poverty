package fr.poverty.spark.kaggle.initialTrain

import fr.poverty.spark.classification.ensembleMethod.adaBoosting.{AdaBoostingLogisticRegressionTask, AdaBoostingNaiveBayesTask}
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
    val metricName = "f1"
    val numberOfWeakClassifierList = List(2, 3, 4) // , 10, 15
    val sourcePath = "src/main/resources"
    val savePath = "submission/adaBoosting"
    val models = List("logisticRegression", "naiveBayes")
    val validationMethod: String = "trainValidation"
    val ratio: Double = 0.5

    // --> features name
    val nullFeatures = Source.fromFile(s"$sourcePath/nullFeaturesNames").getLines.toList.head.split(",")
    val yesNoFeatures = Source.fromFile(s"$sourcePath/yesNoFeaturesNames").getLines.toList.head.split(",")

    // --> Train and Test data set
    val train = new LoadDataSetTask(sourcePath = "data", format = "csv").run(spark, "train")
    val test = new LoadDataSetTask(sourcePath = "data", format = "csv").run(spark, "test")

    val replacementNoneValues = new ReplacementNoneValuesTask(targetColumn, nullFeatures, yesNoFeatures).run(spark, train, test)
    val trainFilled = replacementNoneValues.getTrain
    val testFilled = replacementNoneValues.getTest

    val labelFeatures = new DefineLabelFeaturesTask(idColumn, targetColumn, featureColumn, Array(""), sourcePath).run(spark, trainFilled)
    val labelFeaturesSubmission = new DefineLabelFeaturesTask(idColumn, "", featureColumn, Array(""),sourcePath).run(spark, testFilled)

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
            validationMethod, ratio, metricName)
          logisticRegression.run(spark, labelFeaturesIndexed)
          val prediction = logisticRegression.computePrediction(spark, labelFeaturesIndexed, logisticRegression.getWeakClassifierList, logisticRegression.getWeightWeakClassifierList)
          val submission = logisticRegression.computeSubmission(spark, labelFeaturesSubmission, logisticRegression.getWeakClassifierList, logisticRegression.getWeightWeakClassifierList)
          logisticRegression.savePrediction(indexToString.run(prediction))
          logisticRegression.saveSubmission(indexToString.run(submission))
        } else if(model == "naiveBayes"){
          println(s"Model: $model")
          val naiveBayes = new AdaBoostingNaiveBayesTask(idColumn, labelColumn, featureColumn,
            predictionColumn, weightColumn, numberOfWeakClassifier, s"$savePath/weakClassifier_$numberOfWeakClassifier/$model",
            validationMethod, ratio, metricName, false)
          naiveBayes.run(spark, labelFeaturesIndexed)
          val prediction = naiveBayes.computePrediction(spark, labelFeaturesIndexed, naiveBayes.getWeakClassifierList, naiveBayes.getWeightWeakClassifierList)
          val submission = naiveBayes.computeSubmission(spark, labelFeaturesSubmission, naiveBayes.getWeakClassifierList, naiveBayes.getWeightWeakClassifierList)
          naiveBayes.savePrediction(indexToString.run(prediction))
          naiveBayes.saveSubmission(indexToString.run(submission))
        }
      })
    })



  }
}
