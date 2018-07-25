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
    val defineLabelFeatures = new DefineLabelFeaturesTask(labelColumn, "src/test/resources/featuresNames/featuresNames")
    val features = defineLabelFeatures.readFeatureNames(defineLabelFeatures.getSourcePath)

    assert(features.isInstanceOf[Array[String]])
    assert(features.length == 2)
  }

  @Test def testDefineLabelValues(): Unit = {
    val data = new LoadDataSetTask(sourcePath = "src/test/resources", format="csv")
      .run(spark, "defineLabelFeatures")
    val defineLabelValues = new DefineLabelFeaturesTask(labelColumn, "src/main/resources/featuresNames")
    defineLabelValues.setFeatureNames(Array("x", "y"))
    val labelValues = defineLabelValues.defineLabelValues(spark, data)

    assert(labelValues.isInstanceOf[DataFrame])
    assert(labelValues.columns.length == 2)
    assert(labelValues.columns.contains(labelColumn))
    assert(labelValues.columns.contains("values"))
  }

  @Test def testDefineLabelFeatures(): Unit = {
    val data = new LoadDataSetTask(sourcePath = "src/test/resources", format="csv").run(spark, "defineLabelFeatures")
    val defineLabelFeatures = new DefineLabelFeaturesTask(labelColumn, "src/test/resources/featuresNames/featuresNames")
    val labelFeatures = defineLabelFeatures.run(spark, data)

    assert(labelFeatures.isInstanceOf[DataFrame])
    assert(labelFeatures.columns.length == 2)
    assert(labelFeatures.columns.contains("target"))
    assert(labelFeatures.columns.contains("features"))
    assert(labelFeatures.schema.fields(labelFeatures.schema.fieldIndex("features")).dataType.typeName == "vector")

    // labelFeatures.write.mode("overwrite").parquet("/home/mahjoubi/Documents/github/costa_rican_household_poverty/src/test/resources/classificationTask")
  }

  @After def afterAll() {
    spark.stop()
  }

}



