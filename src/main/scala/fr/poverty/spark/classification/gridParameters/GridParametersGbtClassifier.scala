package fr.poverty.spark.classification.gridParameters

import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.ParamGridBuilder

object GridParametersGbtClassifier {

  def getMaxDepth: Array[Int] = {Array(4, 8, 16, 30)}

  def getMaxBins: Array[Int] = {Array(2, 4, 8, 16)}

  def getParamsGrid(estimator: GBTClassifier): Array[ParamMap] = {
    new ParamGridBuilder()
      .addGrid(estimator.maxDepth, getMaxDepth)
      .addGrid(estimator.maxBins, getMaxBins)
      .build()
  }

}
