package fr.poverty.spark.kaggle

import fr.poverty.spark.classification.trainValidation._
import fr.poverty.spark.utils._
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
//    val trainRatio = 0.50
    val models = Array("decisionTree", "randomForest", "logisticRegression", "oneVsRest", "naiveBayes")
    val sourcePath = "src/main/resources"
//    val savePath = s"submission/trainValidation/trainRatio_${(trainRatio*100).toInt.toString}"

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

    Array(0.50, 0.60, 0.70).foreach(trainRatio => {
      val savePath = s"submission/trainValidation/trainRatio_${(trainRatio*100).toInt.toString}"
      models.foreach(model =>{
        println(s"Model: $model")
        if (model == "decisionTree") {
          val decisionTree = new TrainValidationDecisionTreeTask(labelColumn, featureColumn, predictionColumn, trainRatio, s"$savePath/$model")
          decisionTree.run(labelFeaturesIndexed)
          decisionTree.transform(labelFeaturesSubmission)
          decisionTree.saveSubmission(indexToString.run(decisionTree.getPrediction), idColumn, targetColumn)
        }
        else if (model == "randomForest") {
          val randomForest = new TrainValidationRandomForestTask(labelColumn, featureColumn,predictionColumn, trainRatio,
            s"$savePath/$model")
          randomForest.run(labelFeaturesIndexed)
          randomForest.transform(labelFeaturesSubmission)
          randomForest.saveSubmission(indexToString.run(randomForest.getPrediction), idColumn, targetColumn)
        }
        else if (model == "logisticRegression") {
          val logisticRegression = new TrainValidationLogisticRegressionTask(labelColumn, featureColumn, predictionColumn,
            trainRatio, s"$savePath/$model")
          logisticRegression.run(labelFeaturesIndexed)
          logisticRegression.transform(labelFeaturesSubmission)
          logisticRegression.saveSubmission(indexToString.run(logisticRegression.getPrediction), idColumn, targetColumn)
        }
        else if (model == "oneVsRest") {
          Array("randomForest", "decisionTree", "logisticRegression", "naiveBayes").foreach(classifier => {
            println(s"  Classifier: $classifier")
            val oneVsRest = new TrainValidationOneVsRestTask(labelColumn, featureColumn, predictionColumn,
              trainRatio, s"$savePath/$model/$classifier", classifier, false)
            oneVsRest.run(labelFeaturesIndexed)
            oneVsRest.transform(labelFeaturesSubmission)
            oneVsRest.saveSubmission(indexToString.run(oneVsRest.getPrediction), idColumn, targetColumn)
          })
        }
        else if (model == "naiveBayes") {
          val naiveBayes = new TrainValidationNaiveBayesTask(labelColumn, featureColumn, predictionColumn,
            trainRatio, s"$savePath/$model", false)
          naiveBayes.run(labelFeaturesIndexed)
          naiveBayes.transform(labelFeaturesSubmission)
          naiveBayes.saveSubmission(indexToString.run(naiveBayes.getPrediction), idColumn, targetColumn)}
      })
    }

    )

  }
}
