package fr.poverty.spark.classification.gridParameters

import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.ParamGridBuilder

object GridParametersGbtClassifier {

  def getMaxDepth: Array[Int] = {GridParametersDecisionTree.getMaxDepth}

  def getMaxBins: Array[Int] = {GridParametersDecisionTree.getMaxBins}

  def getImpurity: Array[String] = {GridParametersDecisionTree.getImpurity}

  def getParamsGrid(estimator: GBTClassifier): Array[ParamMap] = {
    new ParamGridBuilder()
      .addGrid(estimator.maxDepth, getMaxDepth)
      .addGrid(estimator.maxBins, getMaxBins)
      .addGrid(estimator.impurity, getImpurity)
      .build()
  }

}
