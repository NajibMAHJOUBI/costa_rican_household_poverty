package fr.poverty.spark.kaggle

import fr.poverty.spark.classification.crossValidation._
import fr.poverty.spark.utils._
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.SparkSession

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
    val numFolds = 5
    val models = Array("decisionTree", "randomForest", "logisticRegression", "oneVsRest", "naiveBayes", "gbtClassifier")
    val sourcePath = "src/main/resources"
    val savePath = "submission/crossValidation/numFolds_5"

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

    val stringIndexer = new StringIndexerTask(targetColumn, labelColumn, savePath)
    val labelFeaturesIndexed = stringIndexer.run(labelFeatures)

    val indexToString = new IndexToStringTask(predictionColumn, targetColumn, stringIndexer.getLabels)

    models.foreach(model =>{
      if (model == "decisionTree") {
        val decisionTree = new CrossValidationDecisionTreeTask(labelColumn, featureColumn, predictionColumn, numFolds, s"$savePath/$model")
        decisionTree.run(labelFeaturesIndexed)
        decisionTree.transform(labelFeaturesSubmission)
        decisionTree.saveSubmission(indexToString.run(decisionTree.getPrediction), idColumn, targetColumn)
      }
      else if (model == "randomForest") {
        val randomForest = new CrossValidationRandomForestTask(labelColumn, featureColumn, predictionColumn, numFolds, s"$savePath/$model")
        randomForest.run(labelFeaturesIndexed)
        randomForest.transform(labelFeaturesSubmission)
        randomForest.saveSubmission(indexToString.run(randomForest.getPrediction), idColumn, targetColumn)
      }
      else if (model == "logisticRegression") {
        val logisticRegression = new CrossValidationLogisticRegressionTask(labelColumn, featureColumn, predictionColumn, numFolds, s"$savePath/$model")
        logisticRegression.run(labelFeaturesIndexed)
        logisticRegression.transform(labelFeaturesSubmission)
        logisticRegression.saveSubmission(indexToString.run(logisticRegression.getPrediction), idColumn, targetColumn)
      }
      else if (model == "oneVsRest") {
        Array("randomForest", "decisionTree", "logisticRegression").foreach(classifier => {
          val oneVsRest = new CrossValidationOneVsRestTask(labelColumn, featureColumn, predictionColumn, numFolds, s"$savePath/$model/$classifier", classifier)
          oneVsRest.run(labelFeaturesIndexed)
          oneVsRest.transform(labelFeaturesSubmission)
          oneVsRest.saveSubmission(indexToString.run(oneVsRest.getPrediction), idColumn, targetColumn)
        })
      }
      else if (model == "naiveBayes") {
        val naiveBayes = new CrossValidationNaiveBayesTask(labelColumn, featureColumn, predictionColumn,
          numFolds, s"$savePath/$model", false)
        naiveBayes.run(labelFeaturesIndexed)
        naiveBayes.transform(labelFeaturesSubmission)
        naiveBayes.saveSubmission(indexToString.run(naiveBayes.getPrediction), idColumn, targetColumn)
      }
      else if(model == "gbtClassifer"){
        val gbtClassifier = new CrossValidationGbtClassifierTask(labelColumn, featureColumn, predictionColumn,
          numFolds, s"$savePath/$model")
        gbtClassifier.run(labelFeaturesIndexed)
        gbtClassifier.transform(labelFeaturesSubmission)
        gbtClassifier.saveSubmission(indexToString.run(gbtClassifier.getPrediction), idColumn, targetColumn)}

})}}
