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

  def mergePredictionWeightList(p: Row, numberOfClassifier: Int): Double = {
    var result: List[(Double, Double)] = List()
    (0 until numberOfClassifier).foreach(index => {
      val elem: mutable.WrappedArray[Double] = p.getAs[mutable.WrappedArray[Double]](p.fieldIndex(s"prediction_$index")).toArray
      result = result ++ List((elem(0), elem(1)))
    })
    // x.groupBy(x => x._1).map(p => (p._1, p._2.map(_._2).reduce(_+_)))
    result.groupBy(_._1).map(p => (p._1, p._2.map(_._2).reduce(_+_))).maxBy(_._2)._1 //.groupBy(x => x._1).map(p => (p._1, p._2.map(_._2).reduce(_+_)))
  }

}
