package fr.poverty.spark.sampleMinorityClass.downMethod

import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType, DoubleType}

/**
  * Created by mahjoubi on 17/08/18.
  *
  * DownSampleMinorityClass test suite
  *
  */

class DownSampleMethodTaskTest extends AssertionsForJUnit  {

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

  @Test def testSmallSize(): Unit = {
    val downMinoritySample = new DownSampleMethodTask(labelColumn)
    val size = downMinoritySample.getSmallSize(downMinoritySample.countByClass(spark, data))
    assert(size.isInstanceOf[Long])
    assert(size == 1L)
  }

  @Test def testResampleTest(): Unit = {
    val downMinoritySample = new DownSampleMethodTask(labelColumn)
    val sample1 = downMinoritySample.resampleClass(spark, data, 1, downMinoritySample.getSmallSize(downMinoritySample.countByClass(spark, data)))
    assert(sample1.isInstanceOf[DataFrame])
    assert(sample1.count() == 1)

    val sample0 = downMinoritySample.resampleClass(spark, data, 0, downMinoritySample.getSmallSize(downMinoritySample.countByClass(spark, data)))
    assert(sample0.isInstanceOf[DataFrame])
    assert(sample0.count() == 1)
  }

  @Test def testResampleDataSet(): Unit = {
    val downMinoritySample = new DownSampleMethodTask(labelColumn)
    val newData = downMinoritySample.resampleDataSet(spark, data, downMinoritySample.countByClass(spark, data), downMinoritySample.getSmallSize(downMinoritySample.countByClass(spark, data)))
    assert(newData.isInstanceOf[DataFrame])
    assert(newData.count() == 2)
  }

  @Test def testRun(): Unit = {
    val downMinoritySample = new DownSampleMethodTask(labelColumn)
    val newData = downMinoritySample.run(spark, data)
    assert(newData.isInstanceOf[DataFrame])
    assert(newData.count() == 2)
  }

  @After def afterAll() {
    spark.stop()
  }

}
