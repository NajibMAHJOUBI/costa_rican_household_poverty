package fr.poverty.spark.utils

import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}

class DefineLabelFeaturesTest {

  private var spark: SparkSession = _
  private val labelColumn = "target"

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
    val features = new DefineLabelFeaturesTask(labelColumn).readFeatureNames()

    assert(features.isInstanceOf[Array[String]])
    assert(features.length == 140)
  }

  @Test def testDefineLabelFeatures(): Unit = {
    val data = new LoadDataSetTask(sourcePath = "src/test/resources", format="csv")
      .run(spark, "defineLabelFeatures")
    data.show()
    data.printSchema()
    val defineLabelValues = new DefineLabelFeaturesTask(labelColumn)
    defineLabelValues.setFeatureNames(Array("x", "y"))
    val labelValues = defineLabelValues.defineLabelValues(spark, data)

    assert(labelValues.isInstanceOf[DataFrame])
    assert(labelValues.columns.length == 2)
    assert(labelValues.columns.contains(labelColumn))
    assert(labelValues.columns.contains("values"))
  }

  @After def afterAll() {
    spark.stop()
  }

}



