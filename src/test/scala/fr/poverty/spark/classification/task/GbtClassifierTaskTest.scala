package fr.poverty.spark.classification.task

import org.apache.log4j.{Level, LogManager}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit

/**
  * Created by mahjoubi on 12/06/18.
  */


class GbtClassifierTaskTest extends AssertionsForJUnit  {

  private val labelColumn: String = "target"
  private val featureColumn: String = "features"
  private val predictionColumn: String = "prediction"
  private var spark: SparkSession = _
  private var data: DataFrame = _

  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("gbt classifier test")
      .getOrCreate()
    
    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)

    val dataSeq = Seq(
      Row("a", 0, new DenseVector(Array(1.0,2.0,1.0,2.0))),
      Row("b", 0, new DenseVector(Array(2.0,4.0,2.0,4.0))),
      Row("c", 1, new DenseVector(Array(1.0,1.0,1.0,1.0))),
      Row("d", 1, new DenseVector(Array(1.0,1.0,1.0,1.0))))
    val rdd = spark.sparkContext.parallelize(dataSeq)
    val schema = StructType(Seq(
      StructField("id", StringType, false),
      StructField(labelColumn, IntegerType, false),
      StructField(featureColumn, VectorType, false)))
    data = spark.createDataFrame(rdd, schema)
  }

  @Test def testGbtClassifier(): Unit = {
    val gbtClassifier = new GbtClassifierTask(labelColumn = "target", featureColumn = "features", predictionColumn = "prediction")
    gbtClassifier.defineEstimator
    gbtClassifier.fit(data)
    val transform = gbtClassifier.transform(data)

    assert(transform.isInstanceOf[DataFrame])
    assert(transform.columns.contains("prediction"))
    assert(transform.columns.contains("probability"))
    assert(transform.columns.contains("rawPrediction"))
  }

  @After def afterAll() {
    spark.stop()
  }
}
