package fr.poverty.spark.classification.task.KNeighborsClassifier

import org.apache.log4j.{Level, LogManager}
import org.apache.spark.ml.feature.MinHashLSHModel
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
  private val inputColumn: String = "features"
  private val outputColumn: String = "hashes"
  private val distanceColumn: String = "jaccardDistance"

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
    val minLSH = new MinHashLSHTask(inputColumn, outputColumn, 5, distanceColumn)
    minLSH.defineEstimator()
    minLSH.fit(dfA)
    assert(minLSH.model.getInputCol == inputColumn)
    assert(minLSH.model.getOutputCol == outputColumn)
    assert(minLSH.model.isInstanceOf[MinHashLSHModel])
  }

  @Test def testApproxSimilarityJoin(): Unit = {
    val minLSH = new MinHashLSHTask(inputColumn, outputColumn, 5, distanceColumn)
    minLSH.defineEstimator()
    minLSH.fit(dfA)
    val similarityJoin = minLSH.approxSimilarityJoin(dfA, dfA)
    assert(similarityJoin.isInstanceOf[DataFrame])
    assert(similarityJoin.columns.length == 3)
    List("idA", "idB", distanceColumn).foreach(column => assert(similarityJoin.columns.contains(column)))
    assert(similarityJoin.count() == math.pow(dfA.count(), 2))
  }




  @After def afterAll() {
    spark.stop()
  }
}
