package fr.poverty.spark.utils

import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{DataFrame, SparkSession}

class DefineStackingLabelFeaturesTask(val idColumn: String, val labelColumn: String, val predictionColumn: String, val mapSourcePath: Map[String, String]) {

  def run(spark: SparkSession, data: DataFrame): Unit = {

  }

  def mergeDataSet(spark: SparkSession): Unit = {

  }

  def loadDataSet(spark: SparkSession, classifier: String, path: String): DataFrame = {
    new LoadDataSetTask("src/test/resources", format = "parquet").run(spark, "prediction")
      .select(col(idColumn), col(predictionColumn).alias(s"${predictionColumn}_${classifier}"))
  }

}
