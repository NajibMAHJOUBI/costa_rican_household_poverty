package fr.poverty.spark.kaggle.initialTrain

import fr.poverty.spark.classification.ensembleMethod.adaBoosting.{AdaBoostingLogisticRegressionTask, AdaBoostingNaiveBayesTask}
import fr.poverty.spark.stat.ChiSquareTask
import fr.poverty.spark.utils._
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.SparkSession

import scala.io.Source

object KaggleChiSquareAdaBoostingExample {

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
    val numberOfWeakClassifierList = List(25, 35, 45, 55) // , 10, 15
    val sourcePath = "src/main/resources"
    val models = List("naiveBayes")
    val validationMethod: String = "trainValidation"
    val alphaValue = 0.05
    val ratio: Double = 0.6
    val saveRootPath: String = s"submission/chiSquare/pValue_${(alphaValue*100).toInt}/$metricName/adaBoosting/$validationMethod/trainRatio_${(ratio*100).toInt.toString}"

    // --> Train and Test data set
    val dataTrain = new LoadDataSetTask(sourcePath = "data", format = "csv").run(spark, "train")
    val dataTest = new LoadDataSetTask(sourcePath = "data", format = "csv").run(spark, "test")

    // --> features name
    val nullFeatures = Source.fromFile(s"$sourcePath/nullFeaturesNames").getLines.toList.head.split(",")
    val yesNoFeatures = Source.fromFile(s"$sourcePath/yesNoFeaturesNames").getLines.toList.head.split(",")
    val replacementNoneValues = new ReplacementNoneValuesTask(targetColumn, nullFeatures, yesNoFeatures).run(spark, dataTrain, dataTest)
    val trainFilled = replacementNoneValues.getTrain
    val testFilled = replacementNoneValues.getTest

    // ChiSquare test + labelFeatures
    val categoricalFeatures = Source.fromFile(s"$sourcePath/categoricalFeatures").getLines.toList.head.split(",").toList
    val chiSquare = new ChiSquareTask(idColumn, targetColumn, categoricalFeatures, featureColumn,0.05)
    chiSquare.run(spark, trainFilled, testFilled)
    val labelFeatures = chiSquare.getLabelFeatures("train")
    val labelFeaturesSubmission = chiSquare.getLabelFeatures("test")

    val stringIndexer = new StringIndexerTask(targetColumn, labelColumn, "")
    val labelFeaturesIndexed = stringIndexer.run(labelFeatures)

    val indexToString = new IndexToStringTask(predictionColumn, targetColumn, stringIndexer.getLabels)

    numberOfWeakClassifierList.foreach(numberOfWeakClassifier => {
      val savePath = s"$saveRootPath/weakClassifier_$numberOfWeakClassifier"
      println(s"number of weak classifier: $numberOfWeakClassifier")
      models.foreach(model => {
        println(s"Model: $model")
        if(model == "logisticRegression"){
          val logisticRegression = new AdaBoostingLogisticRegressionTask(idColumn, labelColumn, featureColumn,
            predictionColumn, weightColumn, numberOfWeakClassifier, s"$savePath/$model",
            validationMethod, ratio, metricName)
          logisticRegression.run(spark, labelFeaturesIndexed)
          val prediction = logisticRegression.computePrediction(spark, labelFeaturesIndexed, logisticRegression.getWeakClassifierList, logisticRegression.getWeightWeakClassifierList)
          val submission = logisticRegression.computeSubmission(spark, labelFeaturesSubmission, logisticRegression.getWeakClassifierList, logisticRegression.getWeightWeakClassifierList)
          logisticRegression.savePrediction(indexToString.run(prediction))
          logisticRegression.saveSubmission(indexToString.run(submission))
        } else if(model == "naiveBayes"){
          val naiveBayes = new AdaBoostingNaiveBayesTask(idColumn, labelColumn, featureColumn,
            predictionColumn, weightColumn, numberOfWeakClassifier, s"$savePath/$model",
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
