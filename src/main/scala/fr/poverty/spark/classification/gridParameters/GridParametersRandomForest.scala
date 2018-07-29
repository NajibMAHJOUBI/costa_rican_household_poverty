package fr.poverty.spark.classification.gridParameters

object GridParametersRandomForest {

  def getMaxDepth: Array[Int] = {Array(4, 8, 16, 30)}

  def getMaxBins: Array[Int] = {Array(2, 4, 8, 16)}

}
