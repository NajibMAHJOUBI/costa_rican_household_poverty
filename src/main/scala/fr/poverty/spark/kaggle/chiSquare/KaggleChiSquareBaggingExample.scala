package fr.poverty.spark.kaggle.chiSquare

import fr.poverty.spark.classification.ensembleMethod.bagging._
import fr.poverty.spark.stat.ChiSquareTask
import fr.poverty.spark.utils._
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.SparkSession

import scala.io.Source

object KaggleChiSquareBaggingExample {

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
    val trainRatioList: List[Double] = List(60.0)
    val sourcePath = "src/main/resources"
    val models = List("naiveBayes")
    val validationMethod: String = "trainValidation"
    val alphaValue = 0.05
    val numberOfSampling = 10
    val saveRootPath: String = s"submission/chiSquare/pValue_${(alphaValue*100).toInt}/$metricName/bagging/$validationMethod"

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

    trainRatioList.foreach(ratio => {
      val savePath = s"$saveRootPath/trainRatio_${ratio.toInt.toString}"
      println(s"ratio: $ratio")
      models.foreach(model => {
        println(s"Model: $model")
        val savePathModel = s"$savePath/$model"
        if(model == "logisticRegression"){
          val logisticRegression = new BaggingLogisticRegressionTask(idColumn, labelColumn, featureColumn,
            predictionColumn, savePathModel, numberOfSampling, 0.75, validationMethod,
            ratio/100.0, metricName)
          logisticRegression.run(labelFeaturesIndexed)
          val prediction = logisticRegression.computePrediction(spark, labelFeaturesIndexed, logisticRegression.getModels)
          val submission = logisticRegression.computeSubmission(spark, labelFeaturesSubmission, logisticRegression.getModels)
          logisticRegression.savePrediction(indexToString.run(prediction))
          logisticRegression.saveSubmission(indexToString.run(submission))
        }
        else if(model == "randomForest"){
          val randomForest = new BaggingRandomForestTask(idColumn, labelColumn, featureColumn,
            predictionColumn, savePathModel, numberOfSampling, 0.75, validationMethod,
            ratio/100.0, metricName)
          randomForest.run(labelFeaturesIndexed)
          val prediction = randomForest.computePrediction(spark, labelFeaturesIndexed, randomForest.getModels)
          val submission = randomForest.computeSubmission(spark, labelFeaturesSubmission, randomForest.getModels)
          randomForest.savePrediction(indexToString.run(prediction))
          randomForest.saveSubmission(indexToString.run(submission))
        }
        else if(model == "decisionTree"){
          val decisionTree = new BaggingDecisionTreeTask(idColumn, labelColumn, featureColumn,
            predictionColumn, savePathModel,
            numberOfSampling, 0.75, validationMethod, ratio/100.0, metricName)
          decisionTree.run(labelFeaturesIndexed)
          val prediction = decisionTree.computePrediction(spark, labelFeaturesIndexed, decisionTree.getModels)
          val submission = decisionTree.computeSubmission(spark, labelFeaturesSubmission, decisionTree.getModels)
          decisionTree.savePrediction(indexToString.run(prediction))
          decisionTree.saveSubmission(indexToString.run(submission))
        }
        else if(model == "naiveBayes"){
          val naiveBayes = new BaggingNaiveBayesTask(idColumn, labelColumn, featureColumn,
            predictionColumn, savePathModel,
            numberOfSampling, 0.75, validationMethod, ratio/100.0, metricName,false)
          naiveBayes.run(labelFeaturesIndexed)
          val prediction = naiveBayes.computePrediction(spark, labelFeaturesIndexed, naiveBayes.getModels)
          val submission = naiveBayes.computeSubmission(spark, labelFeaturesSubmission, naiveBayes.getModels)
          naiveBayes.savePrediction(indexToString.run(prediction))
          naiveBayes.saveSubmission(indexToString.run(submission))
        }
        else if(model == "oneVsRest"){
          models.foreach(classifier => {
            val oneVsRest = new BaggingOneVsRestTask(idColumn, labelColumn, featureColumn,
              predictionColumn, s"$savePathModel/$classifier",
              numberOfSampling, 0.75, validationMethod, ratio/100.0, metricName, classifier)
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
