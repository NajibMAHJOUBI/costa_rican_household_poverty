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
    val dataFilled = replacement.run(spark, data)
    val dataColumns = dataFilled.columns
    assert(dataColumns.length == data.columns.length)
    assert(dataColumns.contains("target"))
    columns.foreach(column => dataColumns.contains(column))
    assert(dataFilled.na.drop(columns).count() == dataFilled.count())
    columns.foreach(column => dataFilled.schema.fields(dataFilled.schema.fieldIndex(column)).dataType == DoubleType)
  }

  @Test def testReplaceMissingYesNoValues(): Unit = {
    val dataYesNo = new LoadDataSetTask(sourcePath = "src/test/resources", format = "csv").run(spark, "replacementYesNoValues")
    val replacement = new ReplacementNoneValuesTask("target", Array(""), columns)
    val dataFilled = replacement.run(spark, dataYesNo)
    val dataColumns = dataFilled.columns
    assert(dataColumns.length == data.columns.length)
    assert(dataColumns.contains("target"))
    columns.foreach(column => dataColumns.contains(column))
    assert(dataFilled.na.drop(columns).count() == dataFilled.count())
    columns.foreach(column => dataFilled.schema.fields(dataFilled.schema.fieldIndex(column)).dataType == DoubleType)

    dataYesNo.show()
    dataFilled.show()
  }

  @After def afterAll() {
    spark.stop()
  }

}



