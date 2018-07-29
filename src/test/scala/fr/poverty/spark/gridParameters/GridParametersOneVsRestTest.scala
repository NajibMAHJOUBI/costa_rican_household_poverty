package fr.poverty.spark.gridParameters

import fr.poverty.spark.classification.gridParameters.GridParametersOneVsRest
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit


/**
  * Created by mahjoubi on 12/06/18.
  */

class GridParametersOneVsRestTest extends AssertionsForJUnit {

  private var spark: SparkSession = _
  private val labelColumn: String = "target"
  private val featureColumn: String = "features"
  private val predictionColumn: String = "prediction"

  @Before def beforeAll() {
    spark = SparkSession.builder.master("local").appName("train validation decision tree task test").getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)
  }

  @Test def testLogisticRegressionGrid(): Unit = {
    val grid = GridParametersOneVsRest.defineLogisticRegressionGrid(labelColumn, featureColumn, predictionColumn)

    assert(grid.isInstanceOf[Array[LogisticRegression]])
    assert(grid.length == 30)
    val logisticRegression = grid(0)
    assert(logisticRegression.getRegParam.isInstanceOf[Double])
    assert(logisticRegression.getElasticNetParam.isInstanceOf[Double])
  }


  @After def afterAll() {
    spark.stop()
  }
}
