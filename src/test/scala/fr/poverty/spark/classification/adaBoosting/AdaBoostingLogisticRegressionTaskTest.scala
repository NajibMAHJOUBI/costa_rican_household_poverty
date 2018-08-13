package fr.poverty.spark.classification.adaBoosting

import fr.poverty.spark.classification.gridParameters.GridParametersLogisticRegression
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


class AdaBoostingLogisticRegressionTaskTest extends AssertionsForJUnit  {

  private val idColumn: String = "id"
  private val labelColumn: String = "target"
  private val featureColumn: String = "features"
  private val predictionColumn: String = "prediction"
  private val weightColumn: String = "weight"
  private val numberOfWeakClassifier: Int = 3
  private val pathSave: String = ""
  private val validationMethod: String = "trainValidation"
  private val ratio: Double = 0.75
  private var adaBoostLR: AdaBoostingLogisticRegressionTask = _
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
    adaBoostLR = new AdaBoostingLogisticRegressionTask(idColumn, labelColumn, featureColumn, predictionColumn, weightColumn, numberOfWeakClassifier, pathSave, validationMethod, ratio)
    adaBoostLR.run(spark, data)
  }

  @Test def testDefineModel(): Unit = {
    val model = adaBoostLR.getModel
    assert(model.getLabelCol == labelColumn)
    assert(model.getFeaturesCol == featureColumn)
    assert(model.getPredictionCol == predictionColumn)
    assert(model.getWeightCol == weightColumn)
  }

  @Test def testWeightWeakClassifierList(): Unit = {
    val weakClassifierList = adaBoostLR.getWeightWeakClassifierList
    assert(weakClassifierList.isInstanceOf[List[Double]])
    assert(weakClassifierList.length == numberOfWeakClassifier)
  }

  @Test def testWeakClassifierList(): Unit = {
    val weightClassifierList = adaBoostLR.getWeakClassifierList
    assert(weightClassifierList.isInstanceOf[List[LogisticRegressionModel]])
    assert(weightClassifierList.length == numberOfWeakClassifier)
  }

  @Test def testComputePrediction(): Unit = {
    val prediction = adaBoostLR.computePrediction(spark, data, adaBoostLR.getWeakClassifierList)
    assert(prediction.isInstanceOf[DataFrame])
    assert(prediction.columns.length == 2)
    assert(prediction.columns.contains(idColumn))
    assert(prediction.columns.contains(predictionColumn))
  }

  @Test def testGridParameters(): Unit = {
    val gridParameters = adaBoostLR.gridParameters()
    assert(gridParameters.isInstanceOf[Array[(Double, Double)]])
    assert(gridParameters.length == GridParametersLogisticRegression.getRegParam.length * GridParametersLogisticRegression.getElasticNetParam.length)
  }

  @After def afterAll() {
    spark.stop()
  }

}
