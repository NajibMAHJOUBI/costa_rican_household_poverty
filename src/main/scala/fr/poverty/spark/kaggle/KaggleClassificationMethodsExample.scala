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


    val replacement = new ReplacementNoneValuesTask("target", Source.fromFile("/home/mahjoubi/Documents/github/costa_rican_household_poverty/src/main/resources/featuresNames").getLines.toList(0).split(","))
    val trainFilled = replacement.computeMeanByColumns(train)

    println(s"Train count: ${trainFilled.count()}")
    println(s"Train drop any rows with nulls: ${trainFilled.na.drop().count()}")


  }
}
