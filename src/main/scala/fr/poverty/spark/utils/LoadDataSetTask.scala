package fr.poverty.spark.utils

import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * Created by mahjoubi on 22/07/18.
  */


class LoadDataSetTask(val sourcePath: String,
                      val format: String) {

  private var data: DataFrame = _

  def run(spark: SparkSession, dataset: String): DataFrame = {
    if (format =="csv") {
      data = spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(s"$sourcePath/$dataset")
    } else {
      data = spark.read
        .parquet(s"$sourcePath/$dataset")
    }
    data
  }

}
