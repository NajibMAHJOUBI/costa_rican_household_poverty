package fr.poverty.spark.sampleMinorityClass.upMethod

import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType, DoubleType}

/**
  * Created by mahjoubi on 17/08/18.
  *
  * UpSampleMinorityClass test suite
  *
  */

class UpSampleMethodTaskTest extends AssertionsForJUnit  {

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

  @Test def testLargeSize(): Unit = {
    val upMinoritySample = new UpSampleMethodTask(labelColumn)
    val size = upMinoritySample.getLargeSize(upMinoritySample.countByClass(spark, data))
    assert(size.isInstanceOf[Long])
    assert(size == 3)
  }

  @Test def testResampleTest(): Unit = {
    val upMinoritySample = new UpSampleMethodTask(labelColumn)
    val sample = upMinoritySample.resampleClass(spark, data, 1, upMinoritySample.getLargeSize(upMinoritySample.countByClass(spark, data)))
    assert(sample.isInstanceOf[DataFrame])
    assert(sample.count() == 3)
  }

  @Test def testResampleDataSet(): Unit = {
    val upMinoritySample = new UpSampleMethodTask(labelColumn)
    val newData = upMinoritySample.resampleDataSet(spark, data, upMinoritySample.countByClass(spark, data), upMinoritySample.getLargeSize(upMinoritySample.countByClass(spark, data)))
    assert(newData.isInstanceOf[DataFrame])
    assert(newData.count() == 6)
  }

  @Test def testRun(): Unit = {
    val upMinoritySample = new UpSampleMethodTask(labelColumn)
    val newData = upMinoritySample.run(spark, data)
    assert(newData.isInstanceOf[DataFrame])
    assert(newData.count() == 6)
  }

  @After def afterAll() {
    spark.stop()
  }

}
