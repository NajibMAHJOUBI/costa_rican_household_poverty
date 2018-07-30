package fr.poverty.spark.classification.gridParameters

import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.ParamGridBuilder

object GridParametersRandomForest {

  def getMaxDepth: Array[Int] = {Array(4, 8, 16, 30)}

  def getMaxBins: Array[Int] = {Array(2, 4, 8, 16)}

  def getParamsGrid(estimator: RandomForestClassifier): Array[ParamMap] = {
    new ParamGridBuilder()
    .addGrid(estimator.maxDepth, getMaxDepth)
    .addGrid(estimator.maxBins, getMaxBins)
    .build()
  }

}
