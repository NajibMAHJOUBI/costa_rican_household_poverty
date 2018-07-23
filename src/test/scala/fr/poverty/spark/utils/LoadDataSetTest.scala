package fr.poverty.spark.utils

import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.SparkSession
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit
import org.apache.spark.sql.types.{DoubleType, IntegerType, StringType}

class LoadDataSetTest {

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

  @Test def testInferSchema(): Unit = {
    val data = new LoadDataSetTask(sourcePath = "src/test/resources", format="csv")
      .run(spark, "loadDataSet")
    assert(data.count() == 2
    )
    val columns = data.columns
    assert(columns.contains("int"))
    assert(columns.contains("float"))
    assert(columns.contains("string"))
    val dataSchema = data.schema
    assert(dataSchema.fields(dataSchema.fieldIndex("int")).dataType == IntegerType)
    assert(dataSchema.fields(dataSchema.fieldIndex("float")).dataType == DoubleType)
    assert(dataSchema.fields(dataSchema.fieldIndex("string")).dataType == StringType)
  }

  @After def afterAll() {
    spark.stop()
  }

}



