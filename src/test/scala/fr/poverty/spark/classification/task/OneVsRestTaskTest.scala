package fr.poverty.spark.classification.task

import fr.poverty.spark.utils.LoadDataSetTask
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit

/**
  * Created by mahjoubi on 12/06/18.
  */
class OneVsRestTaskTest extends AssertionsForJUnit {

  private var spark: SparkSession = _

  @Before def beforeAll() {
    spark = SparkSession.builder.master("local").appName("random forest test").getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)
  }

  @Test def testOneVsRest(): Unit = {
    val data = new LoadDataSetTask("src/test/resources", format = "parquet").run(spark, "classificationTask")
    val oneVsRest = new OneVsRestTask(labelColumn = "target", featureColumn = "features", predictionColumn = "prediction", classifier = "logisticRegression")
    oneVsRest.defineModel
    oneVsRest.fit(data)
    oneVsRest.transform(data)
    val transform = oneVsRest.getPrediction
    assert(transform.isInstanceOf[DataFrame])
    assert(transform.columns.contains("prediction"))
  }

  @After def afterAll() {
    spark.stop()
  }
}
