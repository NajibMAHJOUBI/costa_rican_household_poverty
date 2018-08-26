package fr.poverty.spark.clustering.semiSupervised

import fr.poverty.spark.clustering.task.BisectingKMeansTask
import org.apache.spark.ml.clustering.BisectingKMeansModel
import org.apache.spark.sql.{DataFrame, SparkSession}


class SemiSupervisedBisectingKMeansTask(override val idColumn: String,
                                        override val labelColumn: String,
                                        override val featuresColumn: String,
                                        override val pathSave: String)
  extends SemiSupervisedTask(idColumn, labelColumn, featuresColumn, pathSave)
    with SemiSupervisedFactory {

  override def run(spark: SparkSession, train: DataFrame, test: DataFrame): SemiSupervisedBisectingKMeansTask = {
    defineModel(train)
    computePrediction(train)
    defineMapPredictionLabel(spark, prediction)
    computeSubmission(spark, test)
    savePrediction()
    saveSubmission()
    this
  }

  override def defineModel(data: DataFrame): SemiSupervisedBisectingKMeansTask = {
    val bisectingKMeans = new BisectingKMeansTask(featuresColumn, predictionColumn)
    bisectingKMeans.defineModel(getNumberOfClusters(data))
    bisectingKMeans.fit(data)
    model = bisectingKMeans.model.asInstanceOf[BisectingKMeansModel]
    this
  }

}
