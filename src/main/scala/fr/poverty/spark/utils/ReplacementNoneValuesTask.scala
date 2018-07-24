package fr.poverty.spark.utils

import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.{DoubleType, IntegerType}
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.annotation.switch

class ReplacementNoneValuesTask(val labelColumn: String,
                                val noneColumns: Array[String]) {

  private var meanData: DataFrame = _

  def run(spark: SparkSession, data: DataFrame): DataFrame = {
    meanData = computeMeanByColumns(data)
    replaceMissingValues(spark, data)
  }

  def computeMeanByColumns(data: DataFrame): DataFrame = {
    var meanData = data.groupBy(labelColumn).mean(noneColumns.toSeq: _*)
    noneColumns.foreach(column => meanData = meanData.withColumnRenamed(s"avg($column)", s"$column"))
    meanData
  }

  def replaceMissingValues(spark: SparkSession, data: DataFrame): DataFrame = {
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

}


