package fr.poverty.spark.classification.adaBoosting

import org.apache.spark.sql.Row

import scala.collection.mutable
import scala.math.exp

object AdaBoostingObject {


  def signLabelPrediction(target: Int, prediction: Int): Int = {
    if (target != prediction){1} else {-1}
  }

  def multiplyWeight(weight: Double, diff: Int): Double = diff * weight

  def exponentialWeightObservation(target: Int, prediction: Int, weightClassifier: Double): Double = {
    exp(signLabelPrediction(target, prediction) * weightClassifier)
  }

  def mergePredictionWeight(prediction: Double, weight: Double): List[Double] = List(prediction, weight)

  def mergePredictionWeightList(p: Row, numberOfClassifier: Int): Map[Double, List[mutable.WrappedArray[Double]]] = {
    var result: List[mutable.WrappedArray[Double]] = List()
    (0 until numberOfClassifier).foreach(index => result = result ++ List(p.getAs[mutable.WrappedArray[Double]](p.fieldIndex(s"prediction_$index"))))
    // x.groupBy(x => x._1).map(p => (p._1, p._2.map(_._2).reduce(_+_)))
    result.groupBy(x => x(0)).map(p => (p(0), p(1)))
  }

}
