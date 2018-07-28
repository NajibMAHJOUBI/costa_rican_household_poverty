package fr.poverty.spark.utils

import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}

import scala.collection.Map

class UtilsObjectTest {

  private var spark: SparkSession = _
  private var df: DataFrame = _

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

  @Test def testFillMissedDouble(): Unit = {
    val value0 = UtilsObject.fillMissedDouble(1.0, 1, Map(1 -> 10.0, 2 -> 20.0))
    assert(value0 == 1.0)

    val value1 = UtilsObject.fillMissedDouble(Double.MaxValue, 1, Map(1 -> 10.0, 2 -> 20.0))
    assert(value1 == 10.0)
  }

  @Test def testFillMissedInteger(): Unit = {
    val value0 = UtilsObject.fillMissedInteger(1, 1, Map(1 -> 10.0, 2 -> 20.0))
    assert(value0 == 1.0)

    val value1 = UtilsObject.fillMissedInteger(Integer.MAX_VALUE, 1, Map(1 -> 10.0, 2 -> 20.0))
    assert(value1 == 10.0)
  }

  @After def afterAll() {
    spark.stop()
  }

}



