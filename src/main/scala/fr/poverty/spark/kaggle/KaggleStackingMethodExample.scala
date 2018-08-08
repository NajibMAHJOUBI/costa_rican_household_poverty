package fr.poverty.spark.kaggle

import fr.poverty.spark.classification.stackingMethod.StackingMethodDecisionTreeTask
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.SparkSession

object KaggleStackingMethodExample {

  def main(arguments: Array[String]): Unit = {
    val spark = SparkSession.builder.master("local").appName("Kaggle Submission Example - Stacking Method").getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)

    // --> Initialization
    val idColumn = "Id"
    val targetColumn = "Target"
    val labelColumn = "label"
    val featureColumn = "features"
    val predictionColumn = "prediction"
    val pathTrain = "data"

    val methodValidation = "trainValidation/trainRatio" // "crossValidation/numFolds_"
    val ratio = 50
    val pathPrediction = "submission/"
    var listPathPrediction: List[String] = List("decisionTree", "logisticRegression", "randomForest", "naiveBayes").map(method => s"$pathPrediction/${methodValidation}_$ratio/$method")
    listPathPrediction = listPathPrediction ++  List("decisionTree", "logisticRegression", "randomForest", "naiveBayes").map(method => s"$pathPrediction/${methodValidation}_$ratio/oneVsRest/$method")


    val stackingMethodDecisionTree = new StackingMethodDecisionTreeTask(idColumn, labelColumn, predictionColumn,
      listPathPrediction, "parquet", pathTrain, "csv",
      s"$pathPrediction/${methodValidation}_$ratio/stringIndexer",
      "submission/stackingMethod", "trainValidation", ratio/100.0)


//    override val idColumn: String, override val labelColumn: String, override val predictionColumn: String,
//    override val pathPrediction: List[String], override val formatPrediction: String,
//    override val pathTrain: String, override val formatTrain: String,
//    override val pathStringIndexer: String, override val pathSave: String,
//    override val validationMethod: String, override val ratio: Double



  }

}
