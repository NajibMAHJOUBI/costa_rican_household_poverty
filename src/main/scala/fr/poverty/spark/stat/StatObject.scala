package fr.poverty.spark.stat

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.Row
import scala.collection.mutable.WrappedArray


object StatObject {

  def extractValues(p: Row, columns: List[String]): Array[Double] = {
      var values: Array[Double] = Array()
      columns.foreach(column => values = values :+ p.getInt(p.fieldIndex(column)).toDouble)
      values
    }

  def getMlVector(values: WrappedArray[Double]): Vector = {
      Vectors.dense(values.toArray[Double])
  }

}
