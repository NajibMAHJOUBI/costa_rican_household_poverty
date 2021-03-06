package fr.poverty.spark.classification.ensembleMethod.stackingMethod

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.Row

import scala.collection.mutable.WrappedArray

object StackingMethodObject {

  def extractValues(p: Row, classificationMethods: Array[String]): Array[Double] = {
    var values: Array[Double] = Array()
    (0 to classificationMethods.length - 1).foreach(index => values = values :+ p.getDouble(p.fieldIndex(s"prediction_$index")).toDouble)
    values
  }

  def getMlVector(values: WrappedArray[Double]): Vector = {
    Vectors.dense(values.toArray[Double])
  }
}