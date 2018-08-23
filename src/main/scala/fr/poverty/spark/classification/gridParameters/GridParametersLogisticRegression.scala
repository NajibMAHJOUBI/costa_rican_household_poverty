package fr.poverty.spark.classification.gridParameters

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.ParamGridBuilder

object GridParametersLogisticRegression {

  def getRegParam: Array[Double] = {Array(0.0, 0.001, 0.01, 0.1, 1.0, 10.0)}

  def getElasticNetParam: Array[Double] = {Array(0.0, 0.25, 0.5, 0.75, 1.0)}

  def getFitIntercept: Array[Boolean] = {Array(true, false)}

  def getStandardization: Array[Boolean] = {Array(true, false)}

  def getParamsGrid(estimator: LogisticRegression): Array[ParamMap] = {
    new ParamGridBuilder()
      .addGrid(estimator.regParam, getRegParam)
      .addGrid(estimator.elasticNetParam, getElasticNetParam)
      .addGrid(estimator.fitIntercept, getFitIntercept)
      .addGrid(estimator.standardization, getStandardization)
      .build()
  }

}
