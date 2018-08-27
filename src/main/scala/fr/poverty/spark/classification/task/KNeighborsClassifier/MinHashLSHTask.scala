package fr.poverty.spark.classification.task.KNeighborsClassifier

import org.apache.spark.ml.feature.{MinHashLSH, MinHashLSHModel}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col


class MinHashLSHTask(val inputColumn: String, val outputColumn: String, val numHashTables: Int, val distanceColumn: String) {

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
    this
  }

  def approxSimilarityJoin(dataSetA: DataFrame, dataSetB: DataFrame): DataFrame = {
    model.approxSimilarityJoin(model.transform(dataSetA), model.transform(dataSetB), 1.0, distanceColumn)
      .select(col("datasetA.id").alias("idA"),
        col("datasetB.id").alias("idB"),
        col(distanceColumn))
      .asInstanceOf[DataFrame]
  }

}
