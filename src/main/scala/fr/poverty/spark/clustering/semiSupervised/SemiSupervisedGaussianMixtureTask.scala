 package fr.poverty.spark.clustering.semiSupervised

import fr.poverty.spark.clustering.task.GaussianMixtureTask
import org.apache.spark.ml.clustering.GaussianMixtureModel
import org.apache.spark.sql.{DataFrame, SparkSession}


class SemiSupervisedGaussianMixtureTask(override val idColumn: String,
                                        override val labelColumn: String,
                                        override val featuresColumn: String,
                                        override val pathSave: String)
extends SemiSupervisedTask(idColumn, labelColumn, featuresColumn, pathSave)
  with SemiSupervisedFactory {

  override def run(spark: SparkSession, train: DataFrame, test: DataFrame): SemiSupervisedGaussianMixtureTask = {
    defineModel(train)
    computePrediction(train)
    defineMapPredictionLabel(spark, prediction)
    computeSubmission(spark, test)
    savePrediction()
    saveSubmission()
    this
  }

  override def defineModel(data: DataFrame): SemiSupervisedGaussianMixtureTask = {
    val bisectingKMeans = new GaussianMixtureTask(featuresColumn, predictionColumn)
    bisectingKMeans.defineModel(getNumberOfClusters(data))
    bisectingKMeans.fit(data)
    model = bisectingKMeans.model.asInstanceOf[GaussianMixtureModel]
    this
  }

}
