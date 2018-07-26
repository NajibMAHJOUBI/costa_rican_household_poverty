package fr.poverty.spark.kaggle

import fr.poverty.spark.classification.trainValidation.TrainValidationDecisionTreeTask
import fr.poverty.spark.utils.{DefineLabelFeaturesTask, LoadDataSetTask, ReplacementNoneValuesTask}
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.SparkSession

import scala.io.Source


object KaggleClassificationMethodsExample {

  def main(arguments: Array[String]): Unit = {
    val spark = SparkSession.builder.master("local").appName("Kaggle Submission Example - Classification methods").getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)

    // --> features name
    val nullFeatures = Source.fromFile("src/main/resources/nullFeaturesNames").getLines.toList(0).split(",")
    val yesNoFeatures = Source.fromFile("src/main/resources/yesNoFeaturesNames").getLines.toList(0).split(",")

    // --> Train and Test sata set
    val train = new LoadDataSetTask(sourcePath = "data", format = "csv").run(spark, "train")
    val test = new LoadDataSetTask(sourcePath = "data", format = "csv").run(spark, "test")


    val replacementNoneValues = new ReplacementNoneValuesTask("target", nullFeatures, yesNoFeatures).run(spark, train, test)
    val trainFilled = replacementNoneValues.getTrain
    val testFilled = replacementNoneValues.getTest
    testFilled.show(5)

    val labelFeatures = new DefineLabelFeaturesTask("Id", "Target", "src/main/resources").run(spark, trainFilled)
    val labelFeaturesSubmission = new DefineLabelFeaturesTask("Id", "", "src/main/resources").run(spark, testFilled)

    // --> Model
    val decisionTree = new TrainValidationDecisionTreeTask("Target", "features", "prediction", 0.75, "target/decisionTree")
    decisionTree.run(labelFeatures)
    decisionTree.transform(labelFeatures)
    decisionTree.savePrediction()

    // Submission
    labelFeaturesSubmission.show(5) //    decisionTree.transform(labelFeaturesSubmission)
    //    decisionTree.saveSubmission()


  }
}
