package fr.poverty.spark.clustering.semiSupervised

import fr.poverty.spark.clustering.task.KMeansTask
import org.apache.spark.ml.clustering.KMeansModel
import org.apache.spark.sql.{DataFrame, SparkSession}


class SemiSupervisedKMeansTask(override val idColumn: String,
                               override val labelColumn: String,
                               override val featuresColumn: String,
                               override val pathSave: String)
  extends SemiSupervisedTask(idColumn, labelColumn, featuresColumn, pathSave)
    with SemiSupervisedFactory {

  override def run(spark: SparkSession, train: DataFrame, test: DataFrame): SemiSupervisedKMeansTask = {
    defineModel(train)
    computePrediction(train)
    defineMapPredictionLabel(spark, prediction)
    computeSubmission(spark, test)
    savePrediction()
    saveSubmission()
    this
  }

  override def defineModel(data: DataFrame): SemiSupervisedKMeansTask = {
    val kMeans = new KMeansTask(featuresColumn, predictionColumn)
    kMeans.defineModel(getNumberOfClusters(data))
    kMeans.fit(data)
    model = kMeans.model.asInstanceOf[KMeansModel]
    this
  }

}
