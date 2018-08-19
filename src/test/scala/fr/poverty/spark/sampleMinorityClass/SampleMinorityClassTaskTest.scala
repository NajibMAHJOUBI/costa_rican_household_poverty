package fr.poverty.spark.sampleMinorityClass

import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.types.{DoubleType, IntegerType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit

/**
  * Created by mahjoubi on 17/08/18.
  *
  * SampleMinorityClass test suite
  *
  */

class SampleMinorityClassTaskTest extends AssertionsForJUnit  {

  private val labelColumn: String = "label"
  private var spark: SparkSession = _
  private var data: DataFrame = _

  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("up sample minority class suite test")
      .getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)

    val someData = Seq(Row(0, 10.0, 20.0), Row(0, 40.0, 50.0), Row(0, 70.0, 80.0), Row(1, 0.0, 0.0))
    val schema = StructType(List(StructField("label", IntegerType, false),
      StructField("x", DoubleType, false), StructField("y", DoubleType, false)))
    data = spark.createDataFrame(spark.sparkContext.parallelize(someData), schema)
  }

  @Test def testDataSetClass(): Unit = {
    val sampleMinoritySample = new SampleMinorityClassTask(labelColumn)
    val classList = sampleMinoritySample.dataSetClass(spark, data)
    assert(classList.isInstanceOf[List[Int]])
    assert(classList.length == 2)
  }

  @Test def testCountByClass(): Unit = {
    val sampleMinoritySample = new SampleMinorityClassTask(labelColumn)
    val countClass = sampleMinoritySample.countByClass(spark, data)
    assert(countClass.isInstanceOf[Map[Int, Long]])
    assert(countClass(0) == 3)
    assert(countClass(1) == 1)
  }

  @After def afterAll() {
    spark.stop()
  }

}
