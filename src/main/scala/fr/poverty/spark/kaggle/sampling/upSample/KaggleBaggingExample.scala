package fr.poverty.spark.kaggle.sampling.upSample

import fr.poverty.spark.classification.ensembleMethod.bagging._
import fr.poverty.spark.sampleMinorityClass.upMethod.UpSampleMethodTask
import fr.poverty.spark.utils._
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.SparkSession

import scala.io.Source

object KaggleBaggingExample {

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
    val metricName: String = "f1"
    val trainRatioList = List(60)
    val sourcePath = "src/main/resources"
    val models = List("logisticRegression", "randomForest")
    val validationMethod: String = "trainValidation"
    val savePath = s"submission/sampling/upSample/bagging/$validationMethod"

    // --> features name
    val nullFeatures = Source.fromFile(s"$sourcePath/nullFeaturesNames").getLines.toList.head.split(",")
    val yesNoFeatures = Source.fromFile(s"$sourcePath/yesNoFeaturesNames").getLines.toList.head.split(",")

    // --> Train and Test data set
    val initialTrain = new LoadDataSetTask(sourcePath = "data", format = "csv").run(spark, "train")
    val upMinoritySample = new UpSampleMethodTask(targetColumn)
    val train = upMinoritySample.run(spark, initialTrain)
    val test = new LoadDataSetTask(sourcePath = "data", format = "csv").run(spark, "test")

    val replacementNoneValues = new ReplacementNoneValuesTask(targetColumn, nullFeatures, yesNoFeatures).run(spark, train, test)
    val trainFilled = replacementNoneValues.getTrain
    val testFilled = replacementNoneValues.getTest

    val labelFeatures = new DefineLabelFeaturesTask(idColumn, targetColumn, featureColumn, Array(""), sourcePath).run(spark, trainFilled)
    val labelFeaturesSubmission = new DefineLabelFeaturesTask(idColumn, "", featureColumn, Array(""), sourcePath).run(spark, testFilled)

    val stringIndexer = new StringIndexerTask(targetColumn, labelColumn, "")
    val labelFeaturesIndexed = stringIndexer.run(labelFeatures)

    val indexToString = new IndexToStringTask(predictionColumn, targetColumn, stringIndexer.getLabels)

    trainRatioList.foreach(ratio => {
      println(s"ratio: $ratio")
      (models ++ List("oneVsRest")).foreach(model => {
        println(s"Model: $model")
        if(model == "logisticRegression"){
          val logisticRegression = new BaggingLogisticRegressionTask(idColumn, labelColumn, featureColumn,
            predictionColumn, s"$savePath/trainRatio${ratio.toString}/$model",
            5, 0.75, validationMethod, ratio.toDouble/100.0, metricName)
          logisticRegression.run(labelFeaturesIndexed)
          val prediction = logisticRegression.computePrediction(spark, labelFeaturesIndexed, logisticRegression.getModels)
          val submission = logisticRegression.computeSubmission(spark, labelFeaturesSubmission, logisticRegression.getModels)
          logisticRegression.savePrediction(indexToString.run(prediction))
          logisticRegression.saveSubmission(indexToString.run(submission))
        }
        else if(model == "randomForest"){
          val randomForest = new BaggingRandomForestTask(idColumn, labelColumn, featureColumn,
            predictionColumn, s"$savePath/trainRatio${ratio.toString}/$model",
            5, 0.75, validationMethod, ratio.toDouble/100.0, metricName)
          randomForest.run(labelFeaturesIndexed)
          val prediction = randomForest.computePrediction(spark, labelFeaturesIndexed, randomForest.getModels)
          val submission = randomForest.computeSubmission(spark, labelFeaturesSubmission, randomForest.getModels)
          randomForest.savePrediction(indexToString.run(prediction))
          randomForest.saveSubmission(indexToString.run(submission))
        }
        else if(model == "decisionTree"){
          val decisionTree = new BaggingDecisionTreeTask(idColumn, labelColumn, featureColumn,
            predictionColumn, s"$savePath/trainRatio${ratio.toString}/$model",
            5, 0.75, validationMethod, ratio.toDouble/100.0, metricName)
          decisionTree.run(labelFeaturesIndexed)
          val prediction = decisionTree.computePrediction(spark, labelFeaturesIndexed, decisionTree.getModels)
          val submission = decisionTree.computeSubmission(spark, labelFeaturesSubmission, decisionTree.getModels)
          decisionTree.savePrediction(indexToString.run(prediction))
          decisionTree.saveSubmission(indexToString.run(submission))
        }
        else if(model == "naiveBayes"){
          val naiveBayes = new BaggingNaiveBayesTask(idColumn, labelColumn, featureColumn,
            predictionColumn, s"$savePath/trainRatio${ratio.toString}/$model",
            5, 0.75, validationMethod, ratio.toDouble/100.0, metricName,
            false)
          naiveBayes.run(labelFeaturesIndexed)
          val prediction = naiveBayes.computePrediction(spark, labelFeaturesIndexed, naiveBayes.getModels)
          val submission = naiveBayes.computeSubmission(spark, labelFeaturesSubmission, naiveBayes.getModels)
          naiveBayes.savePrediction(indexToString.run(prediction))
          naiveBayes.saveSubmission(indexToString.run(submission))
        }
        else if(model == "oneVsRest"){
          models.foreach(classifier => {
            val oneVsRest = new BaggingOneVsRestTask(idColumn, labelColumn, featureColumn,
              predictionColumn, s"$savePath/trainRatio${ratio.toString}/$model/$classifier",
              5, 0.75, validationMethod, ratio.toDouble/100.0, metricName,
              classifier)
            oneVsRest.run(labelFeaturesIndexed)
            val prediction = oneVsRest.computePrediction(spark, labelFeaturesIndexed, oneVsRest.getModels)
            val submission = oneVsRest.computeSubmission(spark, labelFeaturesSubmission, oneVsRest.getModels)
            oneVsRest.savePrediction(indexToString.run(prediction))
            oneVsRest.saveSubmission(indexToString.run(submission))
          })

        }
      })
    })



  }

}
