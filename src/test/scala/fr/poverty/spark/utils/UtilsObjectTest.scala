package fr.poverty.spark.utils

import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}

import scala.collection.Map

class UtilsObjectTest {

  private var spark: SparkSession = _
  private var df: DataFrame = _
  private val maps: Map[Int, Double] = Map(100 -> 1.6)

  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("test load dataset")
      .getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)


    val data = Seq((100, 1.5, 3.3), (200, 1.0, 1.0))
    df = spark.createDataFrame(data).toDF("target", "x", "y")
  }

  @Test def testDefineMapMissingValues(): Unit = {
    val map = UtilsObject.defineMapMissingValues(df, "target", "y")

    assert(map.isInstanceOf[Map[Int, Double]])
    assert(map(200) == 1.0)
    assert(map(100) == 3.3)
  }

  @Test def testFillDoubleValue(): Unit = {
    val result = UtilsObject.fillDoubleValue(100, maps)
    assert(result == 1.6)
  }

  @Test def testFillIntValue(): Unit = {
    val result = UtilsObject.fillIntValue(100, maps)
    assert(result == 2)
  }

  @After def afterAll() {
    spark.stop()
  }

}



