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
  private val numberOfWeakClassifier: Int = 3
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

    data = new LoadDataSetTask("src/test/resources", format = "parquet").run(spark, "adaBoost")
    adaBoostLR = new AdaBoostLogisticRegressionTask(idColumn, labelColumn, featureColumn, predictionColumn, weightColumn, numberOfWeakClassifier)
    adaBoostLR.run(spark, data)
  }

  @Test def testDefineModel(): Unit = {
    val model = adaBoostLR.getModel
    assert(model.getLabelCol == labelColumn)
    assert(model.getFeaturesCol == featureColumn)
    assert(model.getPredictionCol == predictionColumn)
    assert(model.getWeightCol == weightColumn)
  }

  @Test def testGetNumberOfObservation(): Unit = {
    data.show()
    assert(adaBoostLR.getNumberOfObservation == 4)
  }

  @Test def testGetNumberOfClass(): Unit = {
    assert(adaBoostLR.getNumberOfClass == 2)
  }

  @Test def testInitialWeights(): Unit = {
    assert(adaBoostLR.getInitialObservationWeight == 0.25)
  }

  @Test def testAddInitialWeightColumn(): Unit = {
    val dataWeight = adaBoostLR.addInitialWeightColumn(data)
    assert(dataWeight.columns.contains(weightColumn))
    assert(dataWeight.select(weightColumn).distinct().rdd.map(p => p.getDouble(p.fieldIndex("weight"))).collect()(0) == 0.25)
  }

  @Test def testSumWeight(): Unit = {
    assert(adaBoostLR.sumWeight(adaBoostLR.addInitialWeightColumn(data)) == 1.0)
  }

  @Test def testWeakClassifierList(): Unit = {
    val weakClassifierList = adaBoostLR.getWeakClassifierList
    assert(weakClassifierList.isInstanceOf[List[Double]])
    assert(weakClassifierList.length == numberOfWeakClassifier)

    weakClassifierList.foreach(println)
  }

//  @Test def testComputeWeightError(): Unit = {
//    val error = adaBoostLR.computeWeightError(new LoadDataSetTask("src/test/resources", format = "parquet").run(spark, "adaBoost"))
//    println(s"Error: $error")
//  }



  @After def afterAll() {
    spark.stop()
  }

}
