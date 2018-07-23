package fr.poverty.spark.utils

import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types.{DoubleType, IntegerType, StringType}
import org.junit.{After, Before, Test}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.udf

class ReplacementNoneValuesTest {

  private var spark: SparkSession = _
  private var data: DataFrame = _

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
    val columns = Array("x", "y")
    val replacement = new ReplacementNoneValuesTask("target", columns)
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
    }

//  @Test def testTargetValues(): Unit = {
//    val targetValues = Array(200, 200)
//    val replacement = new ReplacementNoneValuesTask("target", Array(""))
//    val result = replacement.getTargetValues(data, "target")
//    targetValues.foreach(target => assert(result.contains(target)))
//  }


  def dealNullValue(x: Option[Double]): Option[Double] = {
    val num = x.getOrElse(return None)
    Some(num)
  }


  @Test def defineMap(): Unit = {
      data.show()

      val nullValue = udf((x: Option[Double]) => dealNullValue(x))

      data.withColumn("nullValue", nullValue(col("x"))).show()

//      val columns = Array("x", "y")
//      val replacement = new ReplacementNoneValuesTask("target", columns)
//      replacement.run(data)


  }

  @After def afterAll() {
    spark.stop()
  }

}



