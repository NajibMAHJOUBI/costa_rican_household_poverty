package fr.poverty.spark.kaggle.clustering

import fr.poverty.spark.utils._
import fr.poverty.spark.clustering.semiSupervised.SemiSupervisedKMeansTask
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col

import scala.io.Source


object KaggleClusteringKMeansExample {

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
    val predictionColumn = "prediction"
    val models = List("kMeans")
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

    val labelFeatures = new DefineLabelFeaturesTask(idColumn, targetColumn, continuousFeatures, sourcePath).run(spark, trainFilled)
    val labelFeaturesSubmission = new DefineLabelFeaturesTask(idColumn, "", continuousFeatures, sourcePath).run(spark, testFilled)

    val stringIndexer = new StringIndexerTask(targetColumn, labelColumn, saveRootPath)
    val labelFeaturesIndexed = stringIndexer.run(labelFeatures)
    stringIndexer.saveModel()

    val indexToStringTrain = new IndexToStringTask(predictionColumn, "targetPrediction", stringIndexer.getLabels)
    val indexToStringTest = new IndexToStringTask(predictionColumn, targetColumn, stringIndexer.getLabels)

    models.foreach(model => {
      println(s"Model: $model")
      val savePath = s"$saveRootPath/$model"
        println(s"Model: $model")
        if (model == "kMeans") {
          val kMeans = new SemiSupervisedKMeansTask(idColumn, labelColumn, featureColumn, savePath)
          kMeans.run(spark, labelFeaturesIndexed, labelFeaturesSubmission)
        }
      })
  }
}

