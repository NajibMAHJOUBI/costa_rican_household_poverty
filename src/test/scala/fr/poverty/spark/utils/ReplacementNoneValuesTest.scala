package fr.poverty.spark.utils

import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types.{DoubleType, IntegerType, StringType}
import org.junit.{After, Before, Test}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.udf

class ReplacementNoneValuesTest {

  private var spark: SparkSession = _

  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("test load dataset")
      .getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)
  }

  @Test def testComputeMeanByColumns(): Unit = {
    val data = new LoadDataSetTask(sourcePath = "src/test/resources", format="csv")
      .run(spark, "replacementNoneValues")
    val columns = Array("x", "y")
    val replacement = new ReplacementNoneValuesTask("target", columns)
    val meanData = replacement.computeMeanByColumns(data)

    assert(meanData.isInstanceOf[DataFrame])
    assert(meanData.count() == 2)
    assert(meanData.columns.contains("target"))
    assert(meanData.columns.contains("x"))
    assert(meanData.columns.contains("y"))
    val filterA = udf((x: String) =>  x == "a")
    val meanA = meanData.filter(filterA(col("target"))).rdd.collect()(0)
    assert(meanA(meanA.fieldIndex("target")) == "a")
    assert(meanA(meanA.fieldIndex("x")) == 1.5)
    assert(meanA(meanA.fieldIndex("y")) == 3.0)
    val filterB = udf((x: String) =>  x == "b")
    val meanB = meanData.filter(filterB(col("target"))).rdd.collect()(0)
    assert(meanB(meanB.fieldIndex("target")) == "b")
    assert(meanB(meanB.fieldIndex("x")) == 1.0)
    assert(meanB(meanB.fieldIndex("y")) == 1.0)
    }

  @After def afterAll() {
    spark.stop()
  }

}



