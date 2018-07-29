package fr.poverty.spark.classification.gridParameters

object GridParametersLogisticRegression {

  def getRegParam: Array[Double] = {Array(0.0, 0.001, 0.01, 0.1, 1.0, 10.0)}

  def getElasticNetParam: Array[Double] = {Array(0.0, 0.25, 0.5, 0.75, 1.0)}

}
