package fr.poverty.spark.gridParameters

import fr.poverty.spark.classification.gridParameters.GridParametersDecisionTree
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.SparkSession
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit


/**
  * Created by mahjoubi on 12/06/18.
  */

class GridParametersDecisionTreeTest extends AssertionsForJUnit {

  private var spark: SparkSession = _

  @Before def beforeAll() {
    spark = SparkSession.builder.master("local")
      .appName("test grid parameters decision tree").getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)
  }

  @Test def testGrid(): Unit = {
    val maxBins = GridParametersDecisionTree.getMaxBins
    val maxDepth = GridParametersDecisionTree.getMaxDepth
    val impurity = GridParametersDecisionTree.getImpurity

    assert(maxBins.isInstanceOf[Array[Int]])
    assert(maxDepth.isInstanceOf[Array[Int]])
    assert(impurity.isInstanceOf[Array[String]])
  }


  @After def afterAll() {
    spark.stop()
  }
}
