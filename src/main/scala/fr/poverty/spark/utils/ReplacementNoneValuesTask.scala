package fr.poverty.spark.utils

import org.apache.spark.sql.DataFrame

class ReplacementNoneValuesTask(val labelColumn: String,
                                val noneColumns: Array[String]) {

  def run(data: DataFrame): Unit = {

    val meanData = computeMeanByColumns(data)
    meanData.show()
    meanData.printSchema()

    replaceMissingValues(data, meanData, labelColumn, noneColumns)

  }

  def computeMeanByColumns(data: DataFrame): DataFrame = {
    var meanData = data.groupBy(labelColumn).mean(noneColumns.toSeq: _*)
    noneColumns.foreach(column => meanData = meanData.withColumnRenamed(s"avg($column)", s"$column"))
    meanData
  }

//  def getMeanValue(data: DataFrame, column: String) = {
//    data.filter(col(column) == )
//  }

//  def getTargetValues(data: DataFrame, columnName: String): Array[Int] = {
//    data.select(labelColumn).distinct().rdd.map(x => x.getInt(x.fieldIndex(columnName))).collect()
//  }

  def replaceMissingValues(data: DataFrame, meanData: DataFrame, labelColumn: String, noneColumns: Array[String]) = {
    noneColumns.foreach(column => {
      println(column)
      val map = meanData.rdd.map(x => (x.getInt(x.fieldIndex(labelColumn)), x.getDouble(x.fieldIndex(column)))).collectAsMap()
      println(column, map)
    })




  }

}




