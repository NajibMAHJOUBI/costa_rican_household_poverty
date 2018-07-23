package fr.poverty.spark.utils

import org.apache.spark.sql.DataFrame

class ReplacementNoneValuesTask(val labelColumn: String,
                                val noneColumns: Array[String]) {

  def run(data: DataFrame): Unit = {

  }

  def computeMeanByColumns(data: DataFrame): DataFrame = {
    var meanData = data.groupBy(labelColumn).mean(noneColumns.toSeq: _*)
    noneColumns.foreach(column => meanData = meanData.withColumnRenamed(s"avg($column)", s"$column"))
    meanData
  }

}




