package fr.poverty.spark.utils

import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}

class DefineLabelFeaturesTest {

  private var spark: SparkSession = _
  private val labelColumn = "target"
  private val idColumn = "id"

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
    val defineLabelFeatures = new DefineLabelFeaturesTask(idColumn, labelColumn, "src/test/resources/featuresNames")
    val features = defineLabelFeatures.readFeatureNames(defineLabelFeatures.getSourcePath)

    assert(features.isInstanceOf[Array[String]])
    assert(features.length == 4)
  }

  @Test def testDefineLabelValues(): Unit = {
    val data = new LoadDataSetTask(sourcePath = "src/test/resources", format="csv")
      .run(spark, "defineLabelFeatures")
    val defineLabelValues = new DefineLabelFeaturesTask(idColumn, labelColumn, "src/main/resources/featuresNames")
    defineLabelValues.setFeatureNames(Array("x", "y"))
    val labelValues = defineLabelValues.defineLabelValues(spark, data)

    assert(labelValues.isInstanceOf[DataFrame])
    assert(labelValues.columns.length == 3)
    assert(labelValues.columns.contains(idColumn))
    assert(labelValues.columns.contains(labelColumn))
    assert(labelValues.columns.contains("values"))
  }

  @Test def testDefineLabelFeatures(): Unit = {
    val data = new LoadDataSetTask(sourcePath = "src/test/resources", format="csv").run(spark, "defineLabelFeatures")
    val defineLabelFeatures = new DefineLabelFeaturesTask(idColumn, labelColumn, "src/test/resources/featuresNames")
    val labelFeatures = defineLabelFeatures.run(spark, data)

    assert(labelFeatures.isInstanceOf[DataFrame])
    assert(labelFeatures.columns.length == 3)
    assert(labelFeatures.columns.contains(idColumn))
    assert(labelFeatures.columns.contains(labelColumn))
    assert(labelFeatures.columns.contains("features"))
    assert(labelFeatures.schema.fields(labelFeatures.schema.fieldIndex("features")).dataType.typeName == "vector")
  }

  @After def afterAll() {
    spark.stop()
  }

}



