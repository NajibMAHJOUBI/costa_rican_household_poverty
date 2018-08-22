package fr.poverty.spark.kaggle.initialTrain

import fr.poverty.spark.classification.validation.trainValidation._
import fr.poverty.spark.stat.ChiSquareTask
import fr.poverty.spark.utils._
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.col

import scala.io.Source


object KaggleTrainValidationCategoricalChiSquareExample {

  def main(arguments: Array[String]): Unit = {
    val spark = SparkSession.builder.master("local").appName("Kaggle Submission Example - Classification methods").getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)

    // --> Initialization
    val sourcePath = "src/main/resources"
    val idColumn = "Id"
    val targetColumn = "Target"
    val labelColumn = "label"
    val featureColumn = "features"
    val predictionColumn = "prediction"
    val metricName = "f1"
    val trainRatioList: List[Double] = List(60.0)
    val models = List("decisionTree", "randomForest", "logisticRegression", "naiveBayes") // Array("decisionTree", "randomForest", "logisticRegression", "oneVsRest", "naiveBayes")
    val alphaValue = 0.05
    val saveRootPath: String = s"submission/chiSquare/pValue_${(alphaValue*100).toInt}/$metricName/trainValidation"

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

    val stringIndexer = new StringIndexerTask(targetColumn, labelColumn, saveRootPath)
    val labelFeaturesIndexed = stringIndexer.run(labelFeatures)
    stringIndexer.saveModel()

    val indexToStringTrain = new IndexToStringTask(predictionColumn, "targetPrediction", stringIndexer.getLabels)
    val indexToStringTest = new IndexToStringTask(predictionColumn, targetColumn, stringIndexer.getLabels)

    trainRatioList.foreach(trainRatio => {
      val savePath = s"$saveRootPath/trainRatio_${trainRatio.toInt.toString}"
      (models ++ List("oneVsRest")).foreach(model =>{
        println(s"Model: $model")
        if (model == "decisionTree") {
          val decisionTree = new TrainValidationDecisionTreeTask(labelColumn, featureColumn, predictionColumn, metricName,
            s"$savePath/$model", trainRatio/100.0)
          decisionTree.run(labelFeaturesIndexed)
          decisionTree.saveModel()
          decisionTree.transform(labelFeaturesIndexed)
          decisionTree.savePrediction(indexToStringTrain.run(decisionTree.getPrediction).select(col(idColumn), col("targetPrediction").alias(targetColumn)))
          decisionTree.transform(labelFeaturesSubmission)
          decisionTree.saveSubmission(indexToStringTest.run(decisionTree.getPrediction), idColumn, targetColumn)
        }
        else if (model == "randomForest") {
          val randomForest = new TrainValidationRandomForestTask(labelColumn, featureColumn, predictionColumn, metricName,
            s"$savePath/$model", trainRatio/100.0)
          randomForest.run(labelFeaturesIndexed)
          randomForest.saveModel()
          randomForest.transform(labelFeaturesIndexed)
          randomForest.savePrediction(indexToStringTrain.run(randomForest.getPrediction).select(col(idColumn), col("targetPrediction").alias(targetColumn)))
          randomForest.transform(labelFeaturesSubmission)
          randomForest.saveSubmission(indexToStringTest.run(randomForest.getPrediction), idColumn, targetColumn)
        }
        else if (model == "logisticRegression") {
          val logisticRegression = new TrainValidationLogisticRegressionTask(labelColumn, featureColumn, predictionColumn,
            metricName,s"$savePath/$model", trainRatio/100.0)
          logisticRegression.run(labelFeaturesIndexed)
          logisticRegression.transform(labelFeaturesIndexed)
          logisticRegression.savePrediction(indexToStringTrain.run(logisticRegression.getPrediction).select(col(idColumn), col("targetPrediction").alias(targetColumn)))
          logisticRegression.transform(labelFeaturesSubmission)
          logisticRegression.saveSubmission(indexToStringTest.run(logisticRegression.getPrediction), idColumn, targetColumn)
        }
        else if (model == "oneVsRest") {
          models.foreach(classifier => {
            println(s"  Classifier: $classifier")
            val oneVsRest = new TrainValidationOneVsRestTask(labelColumn, featureColumn, predictionColumn, metricName,
              s"$savePath/$model/$classifier", trainRatio/100.0, classifier, false)
            oneVsRest.run(labelFeaturesIndexed)
            oneVsRest.saveModel()
            oneVsRest.transform(labelFeaturesIndexed)
            oneVsRest.savePrediction(indexToStringTrain.run(oneVsRest.getPrediction).select(col(idColumn), col("targetPrediction").alias(targetColumn)))
            oneVsRest.transform(labelFeaturesSubmission)
            oneVsRest.saveSubmission(indexToStringTest.run(oneVsRest.getPrediction), idColumn, targetColumn)
          })
        }
        else if (model == "naiveBayes") {
          val naiveBayes = new TrainValidationNaiveBayesTask(labelColumn, featureColumn, predictionColumn, metricName,
            s"$savePath/$model", trainRatio/100.0, false)
          naiveBayes.run(labelFeaturesIndexed)
          naiveBayes.transform(labelFeaturesIndexed)
          naiveBayes.savePrediction(indexToStringTrain.run(naiveBayes.getPrediction).select(col(idColumn), col("targetPrediction").alias(targetColumn)))
          naiveBayes.transform(labelFeaturesSubmission)
          naiveBayes.saveSubmission(indexToStringTest.run(naiveBayes.getPrediction), idColumn, targetColumn)}
      })
    })
  }
}

