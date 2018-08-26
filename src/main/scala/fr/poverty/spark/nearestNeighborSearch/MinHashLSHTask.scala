package fr.poverty.spark.nearestNeighborSearch

import org.apache.spark.ml.feature.{MinHashLSH, MinHashLSHModel}
import org.apache.spark.sql.DataFrame

class MinHashLSHTask(val inputColumn: String, val outputColumn: String, val numHashTables: Int) {

  var estimator: MinHashLSH = _
  var model: MinHashLSHModel = _

  def defineEstimator(): MinHashLSHTask = {
    estimator = new MinHashLSH()
        .setNumHashTables(numHashTables)
        .setInputCol(inputColumn)
        .setOutputCol(outputColumn)
    this
  }

  def fit(data: DataFrame): MinHashLSHTask = {
    model = estimator.fit(data)
    model.approxNearestNeighbors()
    this
  }

}
