package fr.poverty.spark.clustering.semiSupervised

import fr.poverty.spark.clustering.task.KMeansTask
import org.apache.spark.ml.Model
import org.apache.spark.ml.clustering.KMeansModel
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.collection.Map


class SemiSupervisedKMeansTask(val idColumn: String,
                               val labelColumn: String,
                               val featuresColumn: String,
                               val pathSave: String) {

  private val predictionColumn: String = "prediction"
  private var model: Model[_] = _
  private var prediction: DataFrame = _
  private var submission: DataFrame = _
  private var mapPredictionLabel: Map[Int, String] = _

  def run(spark: SparkSession, train: DataFrame, test: DataFrame): SemiSupervisedKMeansTask = {
    defineModel(train)
    computePrediction(train)
    defineMapPredictionLabel(prediction)
    computeSubmission(spark, test)
    savePrediction()
    saveSubmission()
    this
  }

  def getNumberOfClusters(data: DataFrame): Int = {
    data.select(labelColumn).distinct().count().toInt
  }

  def defineModel(data: DataFrame): SemiSupervisedKMeansTask = {
    val kMeans = new KMeansTask(featuresColumn, predictionColumn)
    kMeans.defineModel(getNumberOfClusters(data))
    kMeans.fit(data)
    model = kMeans.model.asInstanceOf[KMeansModel]
    this
  }

  def computePrediction(data: DataFrame): SemiSupervisedKMeansTask = {
    prediction = model.transform(data)
    this
  }

  def computeSubmission(spark: SparkSession, data: DataFrame): SemiSupervisedKMeansTask = {
    val mapPredictionLabelBroadcast = spark.sparkContext.broadcast(mapPredictionLabel)
    val getTarget = udf((prediction: Int) => mapPredictionLabelBroadcast.value(prediction))
    submission = model.transform(data)
      .withColumn(labelColumn, getTarget(col(predictionColumn)))
        .select(idColumn, labelColumn)
    this
  }

  def defineMapPredictionLabel(data: DataFrame): SemiSupervisedKMeansTask = {
    mapPredictionLabel = prediction
      .rdd
      .map(row => (row.getInt(row.fieldIndex(predictionColumn)), List(row.getString(row.fieldIndex(labelColumn)))))
      .reduceByKey(_ ++ _)
      .map(p => (p._1, p._2.groupBy(identity).mapValues(_.size).maxBy(_._2)._1))
      .collectAsMap()
    this
  }

  def savePrediction(): SemiSupervisedKMeansTask = {
    prediction
      .coalesce(1)
      .write.mode("overwrite")
      .parquet(s"$pathSave/prediction")
    this
  }

  def saveSubmission(): SemiSupervisedKMeansTask = {
    submission
      .select(col(idColumn), col(labelColumn).cast(IntegerType))
      .repartition(1)
      .write
      .option("header", "true")
      .option("delimiter", ",")
      .mode("overwrite")
      .csv(s"$pathSave/submission")
    this
  }
}
