package fr.poverty.spark.utils

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.DataFrame

import scala.collection.Map
import scala.collection.mutable.WrappedArray


object UtilsObject {

  def defineMapMissingValues(data: DataFrame, labelCol: String, column: String): Map[Int, Double] = {
    data.rdd.map(x => (x.getInt(x.fieldIndex(labelCol)), x.getDouble(x.fieldIndex(column)))).collectAsMap()
  }

  def fillDoubleValue(target: Int, mapTargetValue: Map[Int, Double]): Double = {
    mapTargetValue(target)
  }

  def fillIntValue(target: Int, mapTargetValue: Map[Int, Double]): Double = {
    mapTargetValue(target).round.toDouble
  }

  def defineDenseVector(values: WrappedArray[Double]): Vector = {
    Vectors.dense(values.toArray)
  }
}
