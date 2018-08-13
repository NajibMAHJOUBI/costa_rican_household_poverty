package fr.poverty.spark.classification.gridParameters

import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.ParamGridBuilder

object GridParametersRandomForest {

  def getMaxDepth: Array[Int] = {GridParametersDecisionTree.getMaxDepth}

  def getMaxBins: Array[Int] = {GridParametersDecisionTree.getMaxBins}

  def getImpurity: Array[String] = {GridParametersDecisionTree.getImpurity}

  def getParamsGrid(estimator: RandomForestClassifier): Array[ParamMap] = {
    new ParamGridBuilder()
    .addGrid(estimator.maxDepth, getMaxDepth)
    .addGrid(estimator.maxBins, getMaxBins)
    .addGrid(estimator.impurity, getImpurity)
    .build()
  }

}
