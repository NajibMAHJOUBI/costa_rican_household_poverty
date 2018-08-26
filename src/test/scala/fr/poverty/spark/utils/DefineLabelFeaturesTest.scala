package fr.poverty.spark.utils

import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.junit.{After, Before, Test}

import scala.collection.mutable

class DefineLabelFeaturesTest {

  private var spark: SparkSession = _
  private val featureColumn: String = "features"
  private val labelColumn: String = "target"
  private val idColumn: String = "id"
  private val sourcePath: String = "src/test/resources/featuresNames"
  private var data: DataFrame = _

  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("test load dataset")
      .getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)

    val dataSeq = Seq(
      Row("a", 0, 1.0, 2.0, 3.0),
      Row("b", 0, 4.0, 5.0, 6.0),
      Row("c", 1, 7.0, 8.0, 9.0),
      Row("d", 1, 1.0, 5.0, 9.0))
    val rdd = spark.sparkContext.parallelize(dataSeq)
    val schema = StructType(Seq(
      StructField(idColumn, StringType, false),
      StructField(labelColumn, IntegerType, false),
      StructField("x", DoubleType, false),
      StructField("y", DoubleType, false),
      StructField("z", DoubleType, false)))
    data = spark.createDataFrame(rdd, schema)
  }

  @Test def testReadFeatureNames(): Unit = {
    val defineLabelFeatures = new DefineLabelFeaturesTask(idColumn, labelColumn, featureColumn, Array(""),sourcePath)
    val features = defineLabelFeatures.readFeatureNames()
    assert(features.isInstanceOf[Array[String]])
    assert(features.length == 3)
  }

  @Test def testDefineLabelValues(): Unit = {
    val defineLabelFeatures = new DefineLabelFeaturesTask(idColumn, labelColumn, featureColumn, Array(""), sourcePath)
    val labelValues = defineLabelFeatures.defineLabelValues(spark, data)
    assert(labelValues.isInstanceOf[DataFrame])
    assert(labelValues.columns.length == 3)
    assert(labelValues.columns.contains(idColumn))
    assert(labelValues.columns.contains(labelColumn))
    assert(labelValues.columns.contains("values"))
    val firstRow = labelValues.first()
    assert(firstRow.getAs[mutable.WrappedArray[Double]](firstRow.fieldIndex("values")).length == 3)
  }

  @Test def testDefineLabelValuesWithDrop(): Unit = {
    val defineLabelFeatures = new DefineLabelFeaturesTask(idColumn, labelColumn, featureColumn, Array("z"),sourcePath)
    val labelValues = defineLabelFeatures.defineLabelValues(spark, data)
    assert(labelValues.isInstanceOf[DataFrame])
    assert(labelValues.columns.length == 3)
    assert(labelValues.columns.contains(idColumn))
    assert(labelValues.columns.contains(labelColumn))
    assert(labelValues.columns.contains("values"))
    val firstRow = labelValues.first()
    assert(firstRow.getAs[mutable.WrappedArray[Double]](firstRow.fieldIndex("values")).length == 2)
  }

  @Test def testDefineLabelFeatures(): Unit = {
    val defineLabelFeatures = new DefineLabelFeaturesTask(idColumn, labelColumn, featureColumn, Array(""),sourcePath)
    val labelFeatures = defineLabelFeatures.run(spark, data)
    labelFeatures.show()
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



