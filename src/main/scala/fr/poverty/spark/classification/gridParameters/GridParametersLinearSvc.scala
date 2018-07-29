package fr.poverty.spark.classification.gridParameters

object GridParametersLinearSvc {

  def getRegParam: Array[Double] = {Array(0.0, 0.001, 0.01, 0.1, 1.0, 10.0)}

  def getFitIntercept: Array[Boolean] = {Array(true, false)}

  def getStandardization: Array[Boolean] = {Array(true, false)}
}
