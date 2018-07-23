package fr.poverty.spark.utils

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.udf

class ReplacementNoneValuesTask(val labelColumn: String,
                                val noneColumns: Array[String]) {

  def run(data: DataFrame): Unit = {

    val meanData = computeMeanByColumns(data)
    meanData.show()
    meanData.printSchema()

    replaceMissingValues(data)

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

  def replaceMissingValues(data: DataFrame) = {


    data.show()

//    var dataNullFill: DataFrame = data
//    noneColumns.foreach(column => dataNullFill = dataNullFill.na.fill(Double.NaN, Seq(column)))
//    dataNullFill.show()

    data.na.fill(Double.MaxValue, Seq("x")).show()
    data.na.fill(Int.MaxValue, Seq("y")).show()



//    noneColumns.foreach(column => {
//      println(column)
//      val map = meanData.rdd.map(x => (x.getInt(x.fieldIndex(labelColumn)), x.getDouble(x.fieldIndex(column)))).collectAsMap()
//      println(column, map)
//    })




  }

}




