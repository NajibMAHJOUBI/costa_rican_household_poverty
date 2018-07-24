package fr.poverty.spark.utils

import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.SparkSession
import org.junit.{After, Before, Test}

class DefineLabelFeaturesTest {

  private var spark: SparkSession = _

  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("test load dataset")
      .getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)
  }

  @Test def testInferSchema(): Unit = {
    val features = new DefineLabelFeaturesTask().readFeatureNames()

    assert(features.isInstanceOf[Array[String]])
    assert(features.length == 140)
  }

  @After def afterAll() {
    spark.stop()
  }

}



