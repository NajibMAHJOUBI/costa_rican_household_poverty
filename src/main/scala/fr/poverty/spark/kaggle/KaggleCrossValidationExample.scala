package fr.poverty.spark.kaggle

import fr.poverty.spark.classification.validation.crossValidation._
import fr.poverty.spark.utils._
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col

import scala.io.Source


object KaggleCrossValidationExample {

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
    val models = Array("decisionTree", "randomForest", "logisticRegression", "oneVsRest", "naiveBayes", "gbtClassifier")
    val sourcePath = "src/main/resources"


    // --> features name
    val nullFeatures = Source.fromFile("src/main/resources/nullFeaturesNames").getLines.toList.head.split(",")
    val yesNoFeatures = Source.fromFile("src/main/resources/yesNoFeaturesNames").getLines.toList.head.split(",")

    // --> Train and Test sata set
    val train = new LoadDataSetTask(sourcePath = "data", format = "csv").run(spark, "train")
    val test = new LoadDataSetTask(sourcePath = "data", format = "csv").run(spark, "test")

    val replacementNoneValues = new ReplacementNoneValuesTask(targetColumn, nullFeatures, yesNoFeatures).run(spark, train, test)
    val trainFilled = replacementNoneValues.getTrain
    val testFilled = replacementNoneValues.getTest

    val labelFeatures = new DefineLabelFeaturesTask(idColumn, targetColumn, sourcePath).run(spark, trainFilled)
    val labelFeaturesSubmission = new DefineLabelFeaturesTask(idColumn, "", sourcePath).run(spark, testFilled)

    val stringIndexer = new StringIndexerTask(targetColumn, labelColumn, "")
    val labelFeaturesIndexed = stringIndexer.run(labelFeatures)
    stringIndexer.saveModel()

    val indexToStringTrain = new IndexToStringTask(predictionColumn, "targetPrediction", stringIndexer.getLabels)
    val indexToStringTest = new IndexToStringTask(predictionColumn, targetColumn, stringIndexer.getLabels)

    Array(3,4,5).foreach(numFolds => {
      val savePath = s"submission/crossValidation/numFolds_${numFolds.toString}"
      models.foreach(model =>{
        println(s"Model: $model")
        if (model == "decisionTree") {
          val decisionTree = new CrossValidationDecisionTreeTask(labelColumn, featureColumn, predictionColumn, s"$savePath/$model", numFolds)
          decisionTree.run(labelFeaturesIndexed)
          decisionTree.transform(labelFeaturesIndexed)
          decisionTree.savePrediction(indexToStringTrain.run(decisionTree.getPrediction).select(col(idColumn), col("targetPrediction").alias(targetColumn)))
          decisionTree.transform(labelFeaturesSubmission)
          decisionTree.saveSubmission(indexToStringTest.run(decisionTree.getPrediction), idColumn, targetColumn)
        }
        else if (model == "randomForest") {
          val randomForest = new CrossValidationRandomForestTask(labelColumn, featureColumn, predictionColumn, s"$savePath/$model", numFolds)
          randomForest.run(labelFeaturesIndexed)
          randomForest.transform(labelFeaturesIndexed)
          randomForest.savePrediction(indexToStringTrain.run(randomForest.getPrediction).select(col(idColumn), col("targetPrediction").alias(targetColumn)))
          randomForest.transform(labelFeaturesSubmission)
          randomForest.saveSubmission(indexToStringTest.run(randomForest.getPrediction), idColumn, targetColumn)
        }
        else if (model == "logisticRegression") {
          val logisticRegression = new CrossValidationLogisticRegressionTask(labelColumn, featureColumn, predictionColumn, s"$savePath/$model", numFolds)
          logisticRegression.run(labelFeaturesIndexed)
          logisticRegression.transform(labelFeaturesIndexed)
          logisticRegression.savePrediction(indexToStringTrain.run(logisticRegression.getPrediction).select(col(idColumn), col("targetPrediction").alias(targetColumn)))
          logisticRegression.transform(labelFeaturesSubmission)
          logisticRegression.saveSubmission(indexToStringTest.run(logisticRegression.getPrediction), idColumn, targetColumn)
        }
        else if (model == "oneVsRest") {
          Array("randomForest", "decisionTree", "logisticRegression", "naiveBayes").foreach(classifier => {
            println(s"  Classifier: $classifier")
            val oneVsRest = new CrossValidationOneVsRestTask(labelColumn, featureColumn, predictionColumn, s"$savePath/$model/$classifier", numFolds, classifier)
            oneVsRest.run(labelFeaturesIndexed)
            oneVsRest.transform(labelFeaturesIndexed)
            oneVsRest.savePrediction(indexToStringTrain.run(oneVsRest.getPrediction).select(col(idColumn), col("targetPrediction").alias(targetColumn)))
            oneVsRest.transform(labelFeaturesSubmission)
            oneVsRest.saveSubmission(indexToStringTest.run(oneVsRest.getPrediction), idColumn, targetColumn)
          })
        }
        else if (model == "naiveBayes") {
          val naiveBayes = new CrossValidationNaiveBayesTask(labelColumn, featureColumn, predictionColumn,
            s"$savePath/$model", numFolds, false)
          naiveBayes.run(labelFeaturesIndexed)
          naiveBayes.transform(labelFeaturesIndexed)
          naiveBayes.savePrediction(indexToStringTrain.run(naiveBayes.getPrediction).select(col(idColumn), col("targetPrediction").alias(targetColumn)))
          naiveBayes.transform(labelFeaturesSubmission)
          naiveBayes.saveSubmission(indexToStringTest.run(naiveBayes.getPrediction), idColumn, targetColumn)
        }
      })
    })
    }}
