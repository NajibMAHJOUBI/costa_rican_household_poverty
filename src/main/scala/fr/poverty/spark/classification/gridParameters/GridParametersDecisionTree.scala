package fr.poverty.spark.classification.gridParameters

import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.ParamGridBuilder

object GridParametersDecisionTree {

  def getMaxDepth: Array[Int] = {
    Array(2, 6, 12, 16, 20, 26, 30)
  }

  def getMaxBins: Array[Int] = {
    Array(2)
  }

  def getImpurity: Array[String] = {
    Array("gini", "entropy")
  }

  def getParamsGrid(estimator: DecisionTreeClassifier): Array[ParamMap] = {
    new ParamGridBuilder()
      .addGrid(estimator.maxDepth, getMaxDepth)
      .addGrid(estimator.maxBins, getMaxBins)
      .addGrid(estimator.impurity, getImpurity)
      .build()
  }

}
