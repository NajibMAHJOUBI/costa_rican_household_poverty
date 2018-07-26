package fr.poverty.spark.utils

import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}

class ReplacementNoneValuesTest {

  private var spark: SparkSession = _
  private var data: DataFrame = _
  private val columns: Array[String] = Array("x", "y")

  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("test load dataset")
      .getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)

    data = new LoadDataSetTask(sourcePath = "src/test/resources", format="csv").run(spark, "replacementNoneValues")
  }

  @Test def testComputeMeanByColumns(): Unit = {
    val replacement = new ReplacementNoneValuesTask("target", columns, Array(""))
    val meanData = replacement.computeMeanByColumns(data)

    assert(meanData.isInstanceOf[DataFrame])
    assert(meanData.count() == 2)
    assert(meanData.columns.contains("target"))
    assert(meanData.columns.contains("x"))
    assert(meanData.columns.contains("y"))
    val filterA = udf((x: Int) =>  x == 100)
    val meanA = meanData.filter(filterA(col("target"))).rdd.collect()(0)
    assert(meanA(meanA.fieldIndex("target")) == 100)
    assert(meanA(meanA.fieldIndex("x")) == 1.5)
    assert(meanA(meanA.fieldIndex("y")) == 3.0)
    val filterB = udf((x: Int) =>  x == 200)
    val meanB = meanData.filter(filterB(col("target"))).rdd.collect()(0)
    assert(meanB(meanB.fieldIndex("target")) == 200)
    assert(meanB(meanB.fieldIndex("x")) == 1.0)
    assert(meanB(meanB.fieldIndex("y")) == 1.0)

    data.show()
    meanData.show()
    }

  @Test def testReplaceMissingNullValues(): Unit = {
    val replacement = new ReplacementNoneValuesTask("target", columns, Array(""))
    replacement.run(spark, data, data)

    val train = replacement.getTrain
    val test = replacement.getTest

    train.show()
    test.show()

    val trainColumns = train.columns
    assert(trainColumns.length == data.columns.length)
    assert(trainColumns.contains("target"))
    columns.foreach(column => trainColumns.contains(column))
    assert(train.na.drop(columns).count() == train.count())
    columns.foreach(column => train.schema.fields(train.schema.fieldIndex(column)).dataType == DoubleType)

    val testColumns = test.columns
    assert(testColumns.length == data.columns.length)
    assert(testColumns.contains("target"))
    columns.foreach(column => testColumns.contains(column))
    assert(test.na.drop(columns).count() == test.count())
    columns.foreach(column => test.schema.fields(test.schema.fieldIndex(column)).dataType == DoubleType)
  }

  @Test def testReplaceMissingYesNoValues(): Unit = {
    val dataYesNo = new LoadDataSetTask(sourcePath = "src/test/resources", format = "csv").run(spark, "replacementYesNoValues")
    val replacement = new ReplacementNoneValuesTask("target", Array(""), columns)
    replacement.run(spark, dataYesNo, dataYesNo)
    val train = replacement.getTrain
    val test = replacement.getTest

    val trainColumns = train.columns
    assert(trainColumns.length == data.columns.length)
    assert(trainColumns.contains("target"))
    columns.foreach(column => trainColumns.contains(column))
    assert(train.na.drop(columns).count() == train.count())
    columns.foreach(column => train.schema.fields(train.schema.fieldIndex(column)).dataType == DoubleType)

    val testColumns = test.columns
    assert(testColumns.length == data.columns.length)
    assert(testColumns.contains("target"))
    columns.foreach(column => testColumns.contains(column))
    assert(test.na.drop(columns).count() == test.count())
    columns.foreach(column => test.schema.fields(test.schema.fieldIndex(column)).dataType == DoubleType)
  }

  @After def afterAll() {
    spark.stop()
  }

}



