package fr.poverty.spark.utils

import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.{DoubleType, IntegerType}
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.annotation.switch

class ReplacementNoneValuesTask(val labelColumn: String, val noneColumns: Array[String], val yesNoColumns: Array[String]) {

  var trainFilled: DataFrame = _
  var testFilled: DataFrame = _

  def run(spark: SparkSession, train: DataFrame, test: DataFrame): ReplacementNoneValuesTask = {
    trainFilled = train
    testFilled = test
    replaceMissingValuesByMean(spark)
    replaceYesNoColumns()
    this
  }

  def computeMeanByColumns(data: DataFrame): DataFrame = {
    var meanData = data.groupBy(labelColumn).mean(noneColumns.toSeq: _*)
    noneColumns.foreach(column => meanData = meanData.withColumnRenamed(s"avg($column)", s"$column"))
    meanData
  }

  def replaceMissingValuesByMean(spark: SparkSession): ReplacementNoneValuesTask = {
    if (noneColumns.toList.distinct(0) != "") {
      val meanData = computeMeanByColumns(trainFilled)
      val trainSchema = trainFilled.schema
      noneColumns.foreach(column => {
        val map = UtilsObject.defineMapMissingValues(meanData, labelColumn, column)
        val mapBroadcast = spark.sparkContext.broadcast(map)
        val typeColumn = trainSchema(trainSchema.fieldIndex(column)).dataType
        val valueBroadcast = spark.sparkContext.broadcast(map.values.toArray.sum / map.values.toArray.size.toDouble)

        (typeColumn: @switch) match {
          case DoubleType => {
            trainFilled = trainFilled.na.fill(Double.MaxValue, Seq(column))
            val udfTrainFillMissed = udf((column: Double, target: Int) => UtilsObject.fillMissedDouble(column, target, mapBroadcast.value))
            trainFilled = trainFilled.withColumn(s"filled$column", udfTrainFillMissed(col(column), col(labelColumn)))
            testFilled = testFilled.na.fill(Double.MaxValue, Seq(column))
            val udfTestFillMissed = udf((column: Double) => UtilsObject.fillMissedDouble(column, valueBroadcast.value))
            testFilled = testFilled.withColumn(s"filled$column", udfTestFillMissed(col(column)))
          }
          case IntegerType => {
            trainFilled = trainFilled.na.fill(Int.MaxValue, Seq(column))
            val udfTrainFillMissed = udf((column: Integer, target: Int) => UtilsObject.fillMissedInteger(column, target, mapBroadcast.value))
            trainFilled = trainFilled.withColumn(s"filled$column", udfTrainFillMissed(col(column), col(labelColumn)))
            testFilled = testFilled.na.fill(Int.MaxValue, Seq(column))
            val udfTestFillMissed = udf((column: Integer) => UtilsObject.fillMissedInteger(column, valueBroadcast.value))
            testFilled = testFilled.withColumn(s"filled$column", udfTestFillMissed(col(column)))
          }
        }
      })
      noneColumns.foreach(column => trainFilled = trainFilled.drop(column).withColumnRenamed(s"filled$column", column))
      noneColumns.foreach(column => testFilled = testFilled.drop(column).withColumnRenamed(s"filled$column", column))
      }
    this
  }

  def replaceYesNoColumns(): ReplacementNoneValuesTask = {
    if (yesNoColumns.toList.distinct(0) != "") {
      val udfFillMissed = udf((column: String) => UtilsObject.fillMissedYesNo(column))

      yesNoColumns.foreach(column => trainFilled = trainFilled.withColumn(s"filled$column", udfFillMissed(col(column))))
      yesNoColumns.foreach(column => testFilled = testFilled.withColumn(s"filled$column", udfFillMissed(col(column))))

      yesNoColumns.foreach(column => trainFilled = trainFilled.drop(column).withColumnRenamed(s"filled$column", column))
      yesNoColumns.foreach(column => testFilled = testFilled.drop(column).withColumnRenamed(s"filled$column", column))
    }
    this
  }

  def getTrain: DataFrame = {
    trainFilled
  }

  def getTest: DataFrame = {
    testFilled
  }
}


