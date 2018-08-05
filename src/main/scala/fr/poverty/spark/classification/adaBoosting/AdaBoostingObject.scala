package fr.poverty.spark.classification.adaBoosting

import scala.math.exp

object AdaBoostingObject {


  def signLabelPrediction(target: Int, prediction: Int): Int = {
    if (target != prediction){1} else {-1}
  }

  def multiplyWeight(weight: Double, diff: Int): Double = diff * weight

  def exponentialWeightObservation(target: Int, prediction: Int, weightClassifier: Double): Double = {
    exp(signLabelPrediction(target, prediction) * weightClassifier)
  }



}
