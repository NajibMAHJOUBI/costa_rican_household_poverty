package fr.poverty.spark.classification.adaBoosting

import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.sql.functions.{col, lit, sum, udf}
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.math.log

class AdaBoostingTask(val idColumn: String, val labelColumn: String, val featureColumn: String,
                      val predictionColumn: String, val weightColumn: String,
                      val numberOfWeakClassifier: Int,
                      val pathSave: String) {

  private var numberOfClass: Long = _
  private var numberOfObservation: Long = _
  private var initialObservationWeight: Double = _
  private var weightWeakClassifierList: List[Double] = List()
  private var weakClassifierList: List[LogisticRegressionModel] = List()


  def computePrediction(spark: SparkSession, data: DataFrame): DataFrame = {
    val numberOfClassifier = weightWeakClassifierList.length
    val dataWeight = addUnitaryWeightColumn(data)
    dataWeight.show()
    var idDataSet: DataFrame = data.select(idColumn)
    val numberOfClassifierBroadcast = spark.sparkContext.broadcast(numberOfClassifier)
    val idColumnBroadcast = spark.sparkContext.broadcast(idColumn)
    idDataSet.show()
    (0 until numberOfClassifier).foreach(index => {
      val weightBroadcast = spark.sparkContext.broadcast(weightWeakClassifierList(index))
      val udfMerge = udf((prediction: Double) => AdaBoostingObject.mergePredictionWeight(prediction, weightBroadcast.value))
      val weakTransform = weakClassifierList(index).transform(dataWeight).select(idColumn, predictionColumn)
        .withColumn(s"prediction_$index", udfMerge(col(predictionColumn))).drop(predictionColumn)
        .select(col(idColumn), col(s"prediction_$index"))
      weakTransform.show(5)
      idDataSet.show()
      idDataSet = idDataSet.join(weakTransform, Seq(idColumn))
    })
    idDataSet.show()
    val rdd = idDataSet.rdd.map(p => (p.getString(p.fieldIndex(idColumnBroadcast.value)), AdaBoostingObject.mergePredictionWeightList(p, numberOfClassifierBroadcast.value)))
    spark.createDataFrame(rdd).toDF(idColumn, predictionColumn)
  }

  def computeNumberOfObservation(data: DataFrame): AdaBoostingTask = {
    numberOfObservation = data.count()
    this
  }

  def computeNumberOfClass(data: DataFrame): AdaBoostingTask = {
    numberOfClass = data.select(labelColumn).distinct().count()
    this
  }

  def computeInitialObservationWeight(data: DataFrame): AdaBoostingTask = {
    initialObservationWeight = 1.0 / numberOfObservation
    this
  }

  def addInitialWeightColumn(data: DataFrame): DataFrame = {
    data.withColumn(weightColumn, lit(initialObservationWeight))
  }

  def addUnitaryWeightColumn(data: DataFrame): DataFrame = {
    data.withColumn(weightColumn, lit(1.0))
  }

  def sumWeight(data: DataFrame): Double = {
    val resultSum = if(data.count() == 0){
      0.0
    } else {
      data
        .agg(sum(weightColumn).alias("sum"))
        .rdd
        .map(p => p.getDouble(p.fieldIndex("sum")))
        .collect()(0)
    }
    resultSum
  }

  def computeWeightError(data: DataFrame): Double = {
    val udfFilterLabelPrediction = udf((label: Double, prediction: Double) => label != prediction)
    val sumWeightError = sumWeight(data.filter(udfFilterLabelPrediction(col(labelColumn), col(predictionColumn))))
    val weightSum = sumWeight(data)
    sumWeightError / weightSum
  }

  def computeWeightWeakClassifier(data: DataFrame, error: Double): Double = {
    log((1 - error) / error) + log(numberOfClass - 1)
  }

  def updateWeightObservation(spark: SparkSession, data: DataFrame, weightClassifier: Double): DataFrame = {
    val weightClassifierBroadcast = spark.sparkContext.broadcast(weightClassifier)
    val udfExponentialWeight = udf((target: Int, prediction: Int) => AdaBoostingObject.exponentialWeightObservation(target, prediction, weightClassifierBroadcast.value))
    data.withColumn("updateWeight", udfExponentialWeight(col(labelColumn), col(predictionColumn)))
      .select(col(idColumn), col(labelColumn), col("updateWeight").alias(weightColumn), col(featureColumn))
  }

  def savePrediction(data: DataFrame): AdaBoostingTask = {
    data
      .select(col(idColumn), col("Target").cast(IntegerType))
      .write
      .mode("overwrite")
      .parquet(s"$pathSave/prediction")
    this
  }

  def saveSubmission(data: DataFrame): AdaBoostingTask = {
    data
      .select(col(idColumn), col("Target").cast(IntegerType))
      .repartition(1)
      .write
      .option("header", "true")
      .option("delimiter", ",")
      .mode("overwrite")
      .csv(s"$pathSave/submission")
    this
  }

  def getNumberOfClass: Long = numberOfClass

  def getNumberOfObservation: Long = numberOfObservation

  def getInitialObservationWeight: Double = initialObservationWeight

  def getWeakClassifierList: List[Double] = weightWeakClassifierList
}
