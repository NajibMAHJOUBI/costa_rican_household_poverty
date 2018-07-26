package fr.poverty.spark.utils

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.DataFrame

import scala.collection.Map
import scala.collection.mutable.WrappedArray


object UtilsObject {

  def defineMapMissingValues(data: DataFrame, labelCol: String, column: String): Map[Int, Double] = {
    data.rdd.map(x => (x.getInt(x.fieldIndex(labelCol)), x.getDouble(x.fieldIndex(column)))).collectAsMap()
  }

  def defineDenseVector(values: WrappedArray[Double]): Vector = {
    Vectors.dense(values.toArray)
  }

  def fillMissedDouble(column: Double, target: Int, mapTargetValue: Map[Int, Double]): Double = {
    if (column == Double.MaxValue) {
      mapTargetValue(target)
    } else {
      column
    }
  }

  def fillMissedDouble(column: Double, value: Double): Double = {
    if (column == Double.MaxValue) {
      value
    } else {
      column
    }
  }

  def fillMissedInteger(column: Int, target: Int, mapTargetValue: Map[Int, Double]): Double = {
    if (column == Int.MaxValue) {
      mapTargetValue(target).round.toDouble
    } else {
      column.toDouble
    }
  }

  def fillMissedInteger(column: Int, value: Double): Double = {
    if (column == Int.MaxValue) {
      value.round.toDouble
    } else {
      column.toDouble
    }
  }

  def fillMissedYesNo(column: String): Double = {
    if (column == "yes") {
      1.0
    } else if (column == "no") {
      0.0
    } else {
      column.toDouble
    }
  }

}
