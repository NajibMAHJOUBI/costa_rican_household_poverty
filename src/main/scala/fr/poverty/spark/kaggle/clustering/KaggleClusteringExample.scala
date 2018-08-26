package fr.poverty.spark.kaggle.clustering

import fr.poverty.spark.clustering.semiSupervised.{SemiSupervisedBisectingKMeansTask, SemiSupervisedGaussianMixtureTask, SemiSupervisedKMeansTask}
import fr.poverty.spark.utils._
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.SparkSession

import scala.io.Source


object KaggleClusteringExample {

  def main(arguments: Array[String]): Unit = {
    val spark = SparkSession.builder.master("local").appName("Kaggle Submission Example - Classification methods").getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)

    // --> Initialization
    val sourcePath = "src/main/resources"
    val idColumn = "Id"
    val targetColumn = "Target"
    val labelColumn = "label"
    val featureColumn = "features"
    val models = List("kMeans", "bisectingKMeans", "gaussianMixture")
    val saveRootPath: String = s"submission/clustering"

    // --> Train and Test data set
    val dataTrain = new LoadDataSetTask(sourcePath = "data", format = "csv").run(spark, "train")
    val dataTest = new LoadDataSetTask(sourcePath = "data", format = "csv").run(spark, "test")

    // --> features name
    val nullFeatures = Source.fromFile(s"$sourcePath/nullFeaturesNames").getLines.toList.head.split(",")
    val yesNoFeatures = Source.fromFile(s"$sourcePath/yesNoFeaturesNames").getLines.toList.head.split(",")
    val replacementNoneValues = new ReplacementNoneValuesTask(targetColumn, nullFeatures, yesNoFeatures).run(spark, dataTrain, dataTest)
    val trainFilled = replacementNoneValues.getTrain
    val testFilled = replacementNoneValues.getTest

    // ChiSquare test + labelFeatures
    val continuousFeatures = Source.fromFile(s"$sourcePath/continuousFeatures").getLines.toList.head.split(",")

    val labelFeatures = new DefineLabelFeaturesTask(idColumn, targetColumn, featureColumn, continuousFeatures, sourcePath).run(spark, trainFilled)
    val labelFeaturesSubmission = new DefineLabelFeaturesTask(idColumn, "", featureColumn, continuousFeatures, sourcePath).run(spark, testFilled)

    labelFeatures.show(5)

    labelFeaturesSubmission.show(5)

    models.foreach(model => {
      println(s"Model: $model")
      val savePath = s"$saveRootPath/$model"
        if (model == "kMeans") {
          val kMeans = new SemiSupervisedKMeansTask(idColumn, targetColumn, featureColumn, savePath)
          kMeans.run(spark, labelFeatures, labelFeaturesSubmission)
        } else if (model == "bisectingKMeans") {
        val bisectingKMeans = new SemiSupervisedBisectingKMeansTask(idColumn, targetColumn, featureColumn, savePath)
        bisectingKMeans.run(spark, labelFeatures, labelFeaturesSubmission)
        } else if (model == "gaussianMixture") {
          val gaussianMixture = new SemiSupervisedGaussianMixtureTask(idColumn, targetColumn, featureColumn, savePath)
          gaussianMixture.run(spark, labelFeatures, labelFeaturesSubmission)
        }
      })
  }
}

