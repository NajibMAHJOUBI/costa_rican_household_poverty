package fr.poverty.spark.classification.gridParameters

import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.ParamGridBuilder

object GridParametersLinearSvc {

  def getRegParam: Array[Double] = {Array(0.0, 0.001, 0.01, 0.1, 1.0, 10.0)}

  def getFitIntercept: Array[Boolean] = {Array(true, false)}

  def getStandardization: Array[Boolean] = {Array(true, false)}

  def getParamsGrid(estimator: LinearSVC): Array[ParamMap] = {
    new ParamGridBuilder()
      .addGrid(estimator.regParam, getRegParam)
      .addGrid(estimator.fitIntercept, getFitIntercept)
      .addGrid(estimator.standardization, getStandardization)
      .build()
  }
}
