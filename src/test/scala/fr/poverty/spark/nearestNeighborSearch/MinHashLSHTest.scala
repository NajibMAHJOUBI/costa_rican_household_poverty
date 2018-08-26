package fr.poverty.spark.nearestNeighborSearch

import org.apache.log4j.{Level, LogManager}
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit


/**
  * Created by mahjoubi on 12/06/18.
  */

class MinHashLSHTest extends AssertionsForJUnit {

  private var spark: SparkSession = _
  private var dfA: DataFrame = _

  @Before def beforeAll() {
    spark = SparkSession.builder.master("local")
      .appName("test grid parameters decision tree").getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)

    val data = Seq(
      Row("a", new SparseVector(6, Array(0, 1, 2), Array(1.0, 1.0, 1.0))),
      Row("b", new SparseVector(6, Array(2, 3, 4), Array(1.0, 1.0, 1.0))),
      Row("c", new SparseVector(6, Array(0, 2, 4), Array(1.0, 1.0, 1.0))))
    val rdd = spark.sparkContext.parallelize(data)
    val schema = StructType(Seq(
      StructField("id", StringType, false),
      StructField("features", VectorType, false)))
    dfA = spark.createDataFrame(rdd, schema)
  }

  @Test def testMinLSH(): Unit = {
    val minLSH = new MinHashLSHTask("features", "hashes", 5)
    minLSH.defineEstimator()
    minLSH.fit(dfA)

    minLSH.model.transform(dfA).show()
  }


  @After def afterAll() {
    spark.stop()
  }
}
