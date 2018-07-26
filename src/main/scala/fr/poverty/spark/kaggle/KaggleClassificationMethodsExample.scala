package fr.poverty.spark.kaggle

import fr.poverty.spark.utils.{LoadDataSetTask, ReplacementNoneValuesTask}
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.SparkSession

import scala.io.Source


object KaggleClassificationMethodsExample {

  def main(arguments: Array[String]): Unit = {
    val spark = SparkSession.builder.master("local").appName("Kaggle Submission Example - Classification methods").getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)

    val train = new LoadDataSetTask(sourcePath = "data", format = "csv").run(spark, "train")

    println(s"Train count: ${train.count()}")
    println(s"Train drop any rows with nulls: ${train.na.drop().count()}")


    val nullFeatures = Source.fromFile("src/main/resources/nullFeaturesNames").getLines.toList(0).split(",")
    val yesNoFeatures = Source.fromFile("src/main/resources/yesNoFeaturesNames").getLines.toList(0).split(",")
    val replacement = new ReplacementNoneValuesTask("target", nullFeatures, yesNoFeatures)
    val trainFilled = replacement.computeMeanByColumns(train)

    println(s"Train count: ${trainFilled.count()}")
    println(s"Train drop any rows with nulls: ${trainFilled.na.drop().count()}")


  }
}
