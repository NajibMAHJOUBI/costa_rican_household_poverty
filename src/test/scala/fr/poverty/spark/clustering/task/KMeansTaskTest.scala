package fr.poverty.spark.clustering.task

import org.apache.log4j.{Level, LogManager}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.junit.{After, Before, Test}

class KMeansTaskTest {

  private var spark: SparkSession = _
  private val kClusters: Int = 2
  private val featuresColumn: String = "features"
  private val predictionColumn: String = "prediction"
  private var dataFrame: DataFrame = _

  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("KMeans test suite")
      .getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)

    val data = Seq(new DenseVector(Array(0.0, 0.0)),
      new DenseVector(Array(1.0, 1.0)),
      new DenseVector(Array(9.0, 8.0)),
      new DenseVector(Array(8.0, 9.0))).map(value => Row(value))
    val rdd = spark.sparkContext.parallelize(data)
    val schema = StructType(Seq(StructField("features", VectorType, false)))
    dataFrame = spark.createDataFrame(rdd, schema)
  }

  @Test def testKMeans(): Unit = {
    val kMeans = new KMeansTask(featuresColumn, predictionColumn)
    kMeans.defineModel(kClusters)

    val estimator = kMeans.estimator
    val estimatorParamMap = estimator.extractParamMap()
    assert(estimatorParamMap(estimator.getParam("k")) == kClusters)
    assert(estimatorParamMap(estimator.getParam("featuresCol")) == featuresColumn)
    assert(estimatorParamMap(estimator.getParam("predictionCol")) == predictionColumn)

    kMeans.fit(dataFrame)
    val model = kMeans.model
    val modelParamMap = model.extractParamMap()
    assert(modelParamMap(model.getParam("k")) == kClusters)
    assert(modelParamMap(model.getParam("featuresCol")) == featuresColumn)
    assert(modelParamMap(model.getParam("predictionCol")) == predictionColumn)

    val transform = kMeans.transform(dataFrame)
    assert(transform.isInstanceOf[DataFrame])
    assert(transform.count() == dataFrame.count())
    assert(transform.columns.contains(featuresColumn))
    assert(transform.columns.contains(predictionColumn))
  }

  @After def afterAll() {
    spark.stop()
  }

}



