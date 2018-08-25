package fr.poverty.spark.clustering.task

import org.apache.spark.ml.clustering.GaussianMixture

class GaussianMixtureTask(override val featuresColumn: String, override val predictionColumn: String)
  extends ClusteringTask(featuresColumn, predictionColumn) {

  def defineModel(k: Int): GaussianMixtureTask = {
    estimator = new GaussianMixture()
      .setK(k)
      .setFeaturesCol(featuresColumn)
      .setPredictionCol(predictionColumn)
      .asInstanceOf[GaussianMixture]
    this
  }
}
