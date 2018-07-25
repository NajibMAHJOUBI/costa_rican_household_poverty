package fr.poverty.spark.utils

import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.{DoubleType, IntegerType}
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.annotation.switch

class ReplacementNoneValuesTask(val labelColumn: String, val noneColumns: Array[String], val yesNoColumns: Array[String]) {

  def run(spark: SparkSession, data: DataFrame): DataFrame = {
    var dataFilled: DataFrame = data
    dataFilled = replaceMissingValuesByMean(spark, dataFilled)
    dataFilled = replaceYesNoColumns(dataFilled)
    dataFilled
  }

  def computeMeanByColumns(data: DataFrame): DataFrame = {
    var meanData = data.groupBy(labelColumn).mean(noneColumns.toSeq: _*)
    noneColumns.foreach(column => meanData = meanData.withColumnRenamed(s"avg($column)", s"$column"))
    meanData
  }

  def replaceMissingValuesByMean(spark: SparkSession, data: DataFrame): DataFrame = {
    val meanData = computeMeanByColumns(data)
    var dataFilled: DataFrame = data
    noneColumns.foreach(column => {
      val map = UtilsObject.defineMapMissingValues(meanData, labelColumn, column)

      val mapBroadcast = spark.sparkContext.broadcast(map)

      val schema = data.schema

      val typeColumn = schema(schema.fieldIndex(column)).dataType

      (typeColumn: @switch) match {
        case DoubleType => {
          dataFilled = dataFilled.na.fill(Double.MaxValue, Seq(column))
          val fillMissed = (columns: Double, target: Int) => {
            if (columns == Double.MaxValue) {
              UtilsObject.fillDoubleValue(target, mapBroadcast.value)
            } else {
              columns
            }
          }
          val udfFillMissed = udf(fillMissed)
          dataFilled = dataFilled.withColumn(s"filled$column", udfFillMissed(col(column), col(labelColumn)))
        }
        case IntegerType => {
          dataFilled = dataFilled.na.fill(Double.MaxValue, Seq(column))
          val fillMissed = (columns: Double, target: Int) => {
            if (columns == Int.MaxValue) {
              UtilsObject.fillIntValue(target, mapBroadcast.value)
            } else {
              columns
            }
          }
          val udfFillMissed = udf(fillMissed)
          dataFilled = dataFilled.withColumn(s"filled$column", udfFillMissed(col(column), col(labelColumn)))
        }
      }
    })
    noneColumns.foreach(column => dataFilled = dataFilled.drop(column).withColumnRenamed(s"filled$column", column))
    dataFilled
  }

  def replaceYesNoColumns(data: DataFrame) = {
    var dataFilled: DataFrame = data

    val fillMissed = (column: String) => {
      if (column == "yes") {
        1
      } else if (column == "no") {
        0
      } else {
        column.toInt
      }
    }

    val udfFillMissed = udf(fillMissed)


    yesNoColumns.foreach(column => {
      dataFilled = dataFilled.withColumn(s"filled$column", udfFillMissed(col(column), col(labelColumn)))
    })
    yesNoColumns.foreach(column => dataFilled = dataFilled.drop(column).withColumnRenamed(s"filled$column", column))
    dataFilled
  }

  def replaceDependencyColumn(spark: SparkSession, data: DataFrame): DataFrame = {

    val filterColumn = udf((column: String) => (column == "yes") || (column == "no"))

    val idhogar = data.filter(filterColumn(col("dependency"))).select("idhogar").distinct().rdd.map(p => p.getString(p.fieldIndex("idhogar"))).collect()

    val idhogarBroadcast = spark.sparkContext.broadcast(idhogar)

    val filterIdhogar = udf((idhogar: String) => idhogarBroadcast.value.contains(idhogar))

    val dataFilteredHousehold = data.filter(filterIdhogar(col("idhogar"))).rdd.map(p => (p.getString(p.fieldIndex("idhogar")), Set(p.getInt(p.fieldIndex("age"))))).reduceByKey((x, y) => x ++ y).map(p => (p._1, p._2.filter(p => p < 19 & p > 64).size / p._2.filter(p => p >= 19 & p <= 64).size.toDouble)).collectAsMap()


  }
}


