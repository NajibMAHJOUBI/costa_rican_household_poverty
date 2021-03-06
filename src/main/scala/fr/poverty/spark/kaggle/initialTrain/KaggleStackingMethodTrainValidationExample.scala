package fr.poverty.spark.kaggle.initialTrain

import fr.poverty.spark.classification.ensembleMethod.stackingMethod._
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.SparkSession

object KaggleStackingMethodTrainValidationExample {

  def main(arguments: Array[String]): Unit = {
    val spark = SparkSession.builder.master("local").appName("Kaggle Submission Example - Stacking Method").getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)

    // --> Initialization
    val idColumn = "Id"
    val labelColumn = "Target"
    val predictionColumn = "prediction"
    val metricName: String = "f1"
    val pathTrain = "data"
    val mapFormat = Map("prediction" -> "parquet", "submission" -> "csv")
    val trainRatioList: List[Int] = List(50, 60, 70, 75)
    val methodValidation = "trainValidation" // "crossValidation/numFolds_"
    val pathPrediction = "submission/"
    val models = List("decisionTree", "randomForest", "logisticRegression", "naiveBayes")

    trainRatioList.foreach(ratio => {
      println(s"Ratio: $ratio")
      var listPathPrediction: List[String] = models.map(method => s"$pathPrediction/${methodValidation}/trainRatio_$ratio/$method")
      listPathPrediction = listPathPrediction ++  models.map(method => s"$pathPrediction/${methodValidation}/trainRatio_$ratio/oneVsRest/$method")
      (models ++ List("oneVsRest")).foreach(model =>{
        println(s"Model: $model")
        if (model == "decisionTree") {
          val stackingMethodDecisionTree = new StackingMethodDecisionTreeTask(idColumn, labelColumn, predictionColumn,
            listPathPrediction, mapFormat, pathTrain, "csv",
            s"$pathPrediction/$methodValidation/modelStringIndexer",
            s"submission/stackingMethod/$methodValidation/trainRatio_$ratio/$model", "trainValidation", ratio/100.0, metricName)
          stackingMethodDecisionTree.run(spark)
        } else if(model == "randomForest"){
          val stackingMethodRandomForest = new StackingMethodRandomForestTask(idColumn, labelColumn, predictionColumn,
            listPathPrediction, mapFormat, pathTrain, "csv",
            s"$pathPrediction/$methodValidation/modelStringIndexer",
            s"submission/stackingMethod/$methodValidation/trainRatio_$ratio/$model", "trainValidation", ratio/100.0, metricName)
          stackingMethodRandomForest.run(spark)
        } else if (model == "logisticRegression") {
          val stackingMethodLogisticRegression = new StackingMethodLogisticRegressionTask(idColumn, labelColumn, predictionColumn,
          listPathPrediction, mapFormat, pathTrain, "csv",
          s"$pathPrediction/$methodValidation/modelStringIndexer",
          s"submission/stackingMethod/$methodValidation/trainRatio_$ratio/$model", "trainValidation", ratio/100.0, metricName)
          stackingMethodLogisticRegression.run(spark)
        } else if (model == "oneVsRest"){
          models.foreach(classifier => {
            println(s"Classifier: $classifier")
            val stackingMethodOneVsRest = new StackingMethodOneVsRestTask(idColumn, labelColumn, predictionColumn,
              listPathPrediction, mapFormat, pathTrain, "csv",
              s"$pathPrediction/$methodValidation/modelStringIndexer",
              s"submission/stackingMethod/$methodValidation/trainRatio_$ratio/$model/$classifier", "trainValidation", ratio/100.0, metricName, classifier=classifier, false)
            stackingMethodOneVsRest.run(spark)
          })
        } else if (model == "naiveBayes"){
          val stackingMethodNaiveBayes = new StackingMethodNaiveBayesTask(idColumn, labelColumn, predictionColumn,
            listPathPrediction, mapFormat, pathTrain, "csv",
            s"$pathPrediction/$methodValidation/modelStringIndexer",
            s"submission/stackingMethod/$methodValidation/trainRatio_$ratio/$model", "trainValidation", ratio/100.0, metricName, false)
          stackingMethodNaiveBayes.run(spark)
        }


      })


    })





  }
}
