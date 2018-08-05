package fr.poverty.spark.classification.adaBoosting

import fr.poverty.spark.utils.LoadDataSetTask
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit

/**
  * Created by mahjoubi on 12/06/18.
  */


class AdaBoostingLogisticRegressionTaskTest extends AssertionsForJUnit  {

  private val idColumn: String = "id"
  private val labelColumn: String = "target"
  private val featureColumn: String = "features"
  private val predictionColumn: String = "prediction"
  private val weightColumn: String = "weight"
  private var adaBoostLR: AdaBoostLogisticRegressionTask = _
  private var spark: SparkSession = _
  private var data: DataFrame = _

  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("adaBoosting - logistic regression test")
      .getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)

    data = new LoadDataSetTask("src/test/resources", format = "parquet").run(spark, "classificationTask")
    adaBoostLR = new AdaBoostLogisticRegressionTask(idColumn, labelColumn, featureColumn, predictionColumn, weightColumn)
  }

  @Test def testDefineMModel(): Unit = {
    adaBoostLR.defineModel
    val model = adaBoostLR.getModel
    assert(model.getLabelCol == labelColumn)
    assert(model.getFeaturesCol == featureColumn)
    assert(model.getPredictionCol == predictionColumn)
    assert(model.getWeightCol == weightColumn)
  }

  @Test def testGetNumberOfObservation(): Unit = {
    assert(adaBoostLR.getNumberOfObservation(data) == 4)
  }

  @Test def testGetNumberOfClass(): Unit = {
    assert(adaBoostLR.getNumberOfClass(data) == 2)
  }

  @Test def testInitialWeights(): Unit = {
    assert(adaBoostLR.getInitialWeights(data) == 0.25)
  }

  @Test def testAddInitialWeightColumn(): Unit = {
    val dataWeight = adaBoostLR.addInitialWeightColumn(data)
    assert(dataWeight.columns.contains(weightColumn))
    assert(dataWeight.select(weightColumn).distinct().rdd.map(p => p.getDouble(p.fieldIndex("weight"))).collect()(0) == 0.25)
  }

  @Test def testSumWeight(): Unit = {
    assert(adaBoostLR.sumWeight(adaBoostLR.addInitialWeightColumn(data)) == 1.0)

    assert(adaBoostLR.sumWeight(new LoadDataSetTask("src/test/resources", format = "parquet")
      .run(spark, "adaBoost")) == 1.0)
  }


  @After def afterAll() {
    spark.stop()
  }
}
