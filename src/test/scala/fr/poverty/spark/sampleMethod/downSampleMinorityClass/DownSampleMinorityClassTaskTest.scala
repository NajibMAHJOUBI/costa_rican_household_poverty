package fr.poverty.spark.sampleMethod.upSampleMinorityClass

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

class DownSampleMinorityClassTaskTest extends AssertionsForJUnit  {

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
    val downMinoritySample = new DownSampleMinorityClassTask(labelColumn)
    val classList = downMinoritySample.dataSetClass(spark, data)
    assert(classList.isInstanceOf[List[Int]])
    assert(classList.length == 2)
  }

  @Test def testCountByClass(): Unit = {
    val downMinoritySample = new DownSampleMinorityClassTask(labelColumn)
    val countClass = downMinoritySample.countByClass(spark, data)
    assert(countClass.isInstanceOf[Map[Int, Long]])
    assert(countClass(0) == 3)
    assert(countClass(1) == 1)
  }

  @Test def testSmallSize(): Unit = {
    val downMinoritySample = new DownSampleMinorityClassTask(labelColumn)
    val size = downMinoritySample.getSmallSize(downMinoritySample.countByClass(spark, data))
    assert(size.isInstanceOf[Long])
    assert(size == 3)
  }

  @Test def testResampleTest(): Unit = {
    val downMinoritySample = new DownSampleMinorityClassTask(labelColumn)
    val sample = downMinoritySample.resampleClass(spark, data, 1, downMinoritySample.getSmallSize(downMinoritySample.countByClass(spark, data)))
    assert(sample.isInstanceOf[DataFrame])
    assert(sample.count() == 3)
  }

  @Test def testResampleDataSet(): Unit = {
    val downMinoritySample = new DownSampleMinorityClassTask(labelColumn)
    val newData = downMinoritySample.resampleDataSet(spark, data, downMinoritySample.countByClass(spark, data), downMinoritySample.getSmallSize(downMinoritySample.countByClass(spark, data)))
    assert(newData.isInstanceOf[DataFrame])
    assert(newData.count() == 6)
  }

  @Test def testRun(): Unit = {
    val downMinoritySample = new DownSampleMinorityClassTask(labelColumn)
    val newData = downMinoritySample.run(spark, data)
    assert(newData.isInstanceOf[DataFrame])
    assert(newData.count() == 6)
  }

  @After def afterAll() {
    spark.stop()
  }

}
