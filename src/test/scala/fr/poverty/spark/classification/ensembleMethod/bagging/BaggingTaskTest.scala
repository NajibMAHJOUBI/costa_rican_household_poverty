package fr.poverty.spark.classification.ensembleMethod.bagging

import fr.poverty.spark.utils.LoadDataSetTask
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit

/**
  * Created by mahjoubi on 12/06/18.
  *
  * AdaBoosting test suite
  *
  */


class BaggingTaskTest extends AssertionsForJUnit  {

  private val idColumn: String = "id"
  private val labelColumn: String = "target"
  private val featureColumn: String = "features"
  private val predictionColumn: String = "prediction"
  private val metricName: String = "accuracy"
  private val numberOfSampling: Int = 3
  private val samplingFraction: Double = 0.75
  private val validationMethod: String = "trainValidation"
  private val ratio: Double = 0.70
  private val pathSave: String = ""
  private var spark: SparkSession = _

  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("adaBoosting - suite tests")
      .getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)
  }

  @Test def testBaggingTask(): Unit = {
    val data = new LoadDataSetTask("src/test/resources", format = "parquet").run(spark, "adaBoost")
    val bagging = new BaggingTask(idColumn, labelColumn, featureColumn, predictionColumn, pathSave, numberOfSampling, samplingFraction, validationMethod, ratio, metricName)
    bagging.defineSampleSubset(data)

    val sampleSubsets = bagging.sampleSubsetsList
    assert(sampleSubsets.isInstanceOf[List[DataFrame]])
    assert(sampleSubsets.length == numberOfSampling)
  }

  @After def afterAll() {
    spark.stop()
  }

}
