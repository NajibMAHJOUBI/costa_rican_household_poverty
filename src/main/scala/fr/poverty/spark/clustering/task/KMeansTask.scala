package fr.poverty.spark.clustering.task

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.mllib.clustering.{KMeans => MLlibKMeans}

class KMeansTask(override val featuresColumn: String, override val predictionColumn: String)
  extends ClusteringTask(featuresColumn, predictionColumn) {

  def defineModel(k: Int): KMeansTask = {
    estimator = new KMeans()
      .setK(k)
      .setFeaturesCol(featuresColumn)
      .setPredictionCol(predictionColumn)
      .setInitMode(MLlibKMeans.K_MEANS_PARALLEL)
      .asInstanceOf[KMeans]
    this
  }

}
