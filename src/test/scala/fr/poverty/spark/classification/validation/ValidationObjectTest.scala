package fr.poverty.spark.classification.validation

import fr.poverty.spark.utils.UtilsObject
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.StructType

/**
  * Created by mahjoubi on 12/06/18.
  */

class ValidationObjectTest extends AssertionsForJUnit {

  private var spark: SparkSession = _

  @Before def beforeAll() {

    spark = SparkSession.builder.master("local").appName("validatin object task tests").getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)
  }

  @Test def testTrainValidationSplit(): Unit = {
    val ratio: Double = 0.7
    val data = (1 to 1000).toArray.map(value => Row(value))
    val rdd = spark.sparkContext.parallelize(data)
    val schema = StructType(Seq(StructField(name = "x", dataType = IntegerType, nullable = false)))
    val df: DataFrame = spark.createDataFrame(rdd, schema=schema)

    val dataSplitDF = ValidationObject.trainValidationSplit(df, 0.7)
    val train = dataSplitDF(0)
    val validation = dataSplitDF(1)

    assert(UtilsObject.relativeApproximation(train.count()/df.count().toDouble, ratio, 0.1))
    assert(UtilsObject.relativeApproximation(validation.count()/df.count().toDouble, 1-ratio, 0.1))
  }

  @After def afterAll() {
    spark.stop()
  }
}
