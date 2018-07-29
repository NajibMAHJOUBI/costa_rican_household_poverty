package fr.poverty.spark.utils

import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types.{DoubleType, IntegerType, StringType}
import org.junit.{After, Before, Test}

class StringIndexerTaskTest {

  private var spark: SparkSession = _

  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("test string indexer")
      .getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)
  }

  @Test def testStringIndexer(): Unit = {
    val data = new LoadDataSetTask(sourcePath = "src/test/resources", format="csv")
      .run(spark, "stringIndexer")
    val stringIndexer = new StringIndexerTask("target", "label", "target/stringIndexer")
    val indexed = stringIndexer.run(data)
    assert(indexed.isInstanceOf[DataFrame])
    assert(indexed.columns.length == 2)
    assert(indexed.columns.contains("label"))
    assert(indexed.select("label").distinct().count() == indexed.select("target").distinct().count())

    val indexToString = new IndexToStringTask("label", "Target", stringIndexer.getLabels)
    val reIndexed = indexToString.run(data)
    assert(reIndexed.isInstanceOf[DataFrame])
    assert(reIndexed.columns.length == 3)
    assert(reIndexed.columns.contains("Target"))
    assert(reIndexed.select("Target").distinct().count() == reIndexed.select("label").distinct().count())
  }

  @After def afterAll() {
    spark.stop()
  }

}



