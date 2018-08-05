package fr.poverty.spark.kaggle

import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.SparkSession

object KaggleSubmisionStackingMethodExample {

  def main(arguments: Array[String]): Unit = {
    val spark = SparkSession.builder.master("local").appName("Kaggle Submission Example - Stacking Method").getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)


  }

}
