package fr.poverty.spark.clustering.task

import org.apache.spark.ml.clustering.BisectingKMeans

class BisectingKMeansTask(override val featuresColumn: String, override val predictionColumn: String)
  extends ClusteringTask(featuresColumn, predictionColumn) {

  def defineModel(k: Int): BisectingKMeansTask = {
    estimator = new BisectingKMeans()
      .setK(k)
      .setFeaturesCol(featuresColumn)
      .setPredictionCol(predictionColumn)
      .asInstanceOf[BisectingKMeans]
    this
  }

}
