package fr.poverty.spark.kaggle.sampling.downSample

import fr.poverty.spark.classification.validation.trainValidation._
import fr.poverty.spark.sampleMinorityClass.downMethod.DownSampleMethodTask
import fr.poverty.spark.utils._
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col

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
    val models = Array( "randomForest", "logisticRegression", "oneVsRest") // Array("decisionTree", "randomForest", "logisticRegression",  "naiveBayes", "oneVsRest")
    val trainRatioList = List(0.60) // List(0.50, 0.60, 0.70, 0.75)
    val sourcePath = "src/main/resources"

    // --> features name
    val nullFeatures = Source.fromFile(s"$sourcePath/nullFeaturesNames").getLines.toList.head.split(",")
    val yesNoFeatures = Source.fromFile(s"$sourcePath/yesNoFeaturesNames").getLines.toList.head.split(",")

    // --> Train and Test data set
    val initialTrain = new LoadDataSetTask(sourcePath = "data", format = "csv").run(spark, "train")
    val downMinoritySample = new DownSampleMethodTask(targetColumn)
    val train = downMinoritySample.run(spark, initialTrain)
    val test = new LoadDataSetTask(sourcePath = "data", format = "csv").run(spark, "test")

    val replacementNoneValues = new ReplacementNoneValuesTask(targetColumn, nullFeatures, yesNoFeatures).run(spark, train, test)
    val trainFilled = replacementNoneValues.getTrain
    val testFilled = replacementNoneValues.getTest

    val labelFeatures = new DefineLabelFeaturesTask(idColumn, targetColumn, sourcePath).run(spark, trainFilled)
    val labelFeaturesSubmission = new DefineLabelFeaturesTask(idColumn, "", sourcePath).run(spark, testFilled)

    val stringIndexer = new StringIndexerTask(targetColumn, labelColumn, "submission/upSample/trainValidation")
    val labelFeaturesIndexed = stringIndexer.run(labelFeatures)
    stringIndexer.saveModel()

    val indexToStringTrain = new IndexToStringTask(predictionColumn, "targetPrediction", stringIndexer.getLabels)
    val indexToStringTest = new IndexToStringTask(predictionColumn, targetColumn, stringIndexer.getLabels)

    trainRatioList.foreach(trainRatio => {
      val savePath = s"submission/upSample/trainValidation/trainRatio_${(trainRatio*100).toInt.toString}"
      models.foreach(model =>{
        println(s"Model: $model")
        if (model == "decisionTree") {
          val decisionTree = new TrainValidationDecisionTreeTask(labelColumn, featureColumn, predictionColumn, s"$savePath/$model", trainRatio)
          decisionTree.run(labelFeaturesIndexed)
          decisionTree.saveModel()
          decisionTree.transform(labelFeaturesIndexed)
          decisionTree.savePrediction(indexToStringTrain.run(decisionTree.getPrediction).select(col(idColumn), col("targetPrediction").alias(targetColumn)))
          decisionTree.transform(labelFeaturesSubmission)
          decisionTree.saveSubmission(indexToStringTest.run(decisionTree.getPrediction), idColumn, targetColumn)
        }
        else if (model == "randomForest") {
          val randomForest = new TrainValidationRandomForestTask(labelColumn, featureColumn,predictionColumn,
            s"$savePath/$model", trainRatio)
          randomForest.run(labelFeaturesIndexed)
          randomForest.saveModel()
          randomForest.transform(labelFeaturesIndexed)
          randomForest.savePrediction(indexToStringTrain.run(randomForest.getPrediction).select(col(idColumn), col("targetPrediction").alias(targetColumn)))
          randomForest.transform(labelFeaturesSubmission)
          randomForest.saveSubmission(indexToStringTest.run(randomForest.getPrediction), idColumn, targetColumn)
        }
        else if (model == "logisticRegression") {
          val logisticRegression = new TrainValidationLogisticRegressionTask(labelColumn, featureColumn,
            predictionColumn, s"$savePath/$model", trainRatio)
          logisticRegression.run(labelFeaturesIndexed)
          logisticRegression.transform(labelFeaturesIndexed)
          logisticRegression.savePrediction(indexToStringTrain.run(logisticRegression.getPrediction).select(col(idColumn), col("targetPrediction").alias(targetColumn)))
          logisticRegression.transform(labelFeaturesSubmission)
          logisticRegression.saveSubmission(indexToStringTest.run(logisticRegression.getPrediction), idColumn, targetColumn)
        }
        else if (model == "oneVsRest") {
          Array("randomForest").foreach(classifier => {
            println(s"  Classifier: $classifier")
            val oneVsRest = new TrainValidationOneVsRestTask(labelColumn, featureColumn, predictionColumn,
              s"$savePath/$model/$classifier", trainRatio, classifier, false)
            oneVsRest.run(labelFeaturesIndexed)
            oneVsRest.saveModel()
            oneVsRest.transform(labelFeaturesIndexed)
            oneVsRest.savePrediction(indexToStringTrain.run(oneVsRest.getPrediction).select(col(idColumn), col("targetPrediction").alias(targetColumn)))
            oneVsRest.transform(labelFeaturesSubmission)
            oneVsRest.saveSubmission(indexToStringTest.run(oneVsRest.getPrediction), idColumn, targetColumn)
          })
        }
        else if (model == "naiveBayes") {
          val naiveBayes = new TrainValidationNaiveBayesTask(labelColumn, featureColumn, predictionColumn,
            s"$savePath/$model", trainRatio, false)
          naiveBayes.run(labelFeaturesIndexed)
          naiveBayes.transform(labelFeaturesIndexed)
          naiveBayes.savePrediction(indexToStringTrain.run(naiveBayes.getPrediction).select(col(idColumn), col("targetPrediction").alias(targetColumn)))
          naiveBayes.transform(labelFeaturesSubmission)
          naiveBayes.saveSubmission(indexToStringTest.run(naiveBayes.getPrediction), idColumn, targetColumn)}
      })
    })
  }

}
