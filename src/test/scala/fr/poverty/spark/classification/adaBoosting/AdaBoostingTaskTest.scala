package fr.poverty.spark.classification.adaBoosting

import fr.poverty.spark.utils.LoadDataSetTask
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit

/**
  * Created by mahjoubi on 12/06/18.
  *
  * AdaBoosting test suite
  *
  */


class AdaBoostingTaskTest extends AssertionsForJUnit  {

  private val idColumn: String = "id"
  private val labelColumn: String = "target"
  private val featureColumn: String = "features"
  private val predictionColumn: String = "prediction"
  private val weightColumn: String = "weight"
  private val numberOfWeakClassifier: Int = 3
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

  @Test def testAdaBoostingTask(): Unit = {
    val data = new LoadDataSetTask("src/test/resources", format = "parquet").run(spark, "adaBoost")
    val adaBoosting = new AdaBoostingTask(idColumn, labelColumn, featureColumn, predictionColumn, weightColumn, numberOfWeakClassifier, pathSave, validationMethod, ratio)
    adaBoosting.computeNumberOfObservation(data)
    adaBoosting.computeNumberOfClass(data)
    adaBoosting.computeInitialObservationWeight(data)

    assert(adaBoosting.getNumberOfObservation == 4)
    assert(adaBoosting.getNumberOfClass == 3L)
    assert(adaBoosting.getInitialObservationWeight == 0.25)
    assert(adaBoosting.sumWeight(adaBoosting.addInitialWeightColumn(data)) == 1.0)
    assert(adaBoosting.getWeightWeakClassifierList.isInstanceOf[List[Double]])

    val dataWeight = adaBoosting.addInitialWeightColumn(data)
    assert(dataWeight.isInstanceOf[DataFrame])
    assert(dataWeight.columns.contains(weightColumn))
    assert(dataWeight.select(weightColumn).distinct().rdd.map(p => p.getDouble(p.fieldIndex("weight"))).collect()(0) == 0.25)
  }

  @After def afterAll() {
    spark.stop()
  }

}
