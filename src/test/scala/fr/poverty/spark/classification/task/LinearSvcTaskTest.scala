package fr.poverty.spark.classification.task

import fr.poverty.spark.utils.LoadDataSetTask
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit

/**
  * Created by mahjoubi on 12/06/18.
  */


class LinearSvcTaskTest extends AssertionsForJUnit  {

  private var spark: SparkSession = _

  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("gbt classifier test")
      .getOrCreate()
    
    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)
  }

  @Test def testLinearSvc(): Unit = {
    val data = new LoadDataSetTask("src/test/resources", format = "parquet").run(spark, "classificationTask")
    val linearSvc = new LinearSvcTask(labelColumn = "target", featureColumn = "features", predictionColumn = "prediction")
    linearSvc.defineModel
    linearSvc.fit(data)
    linearSvc.transform(data)
    val transform = linearSvc.getPrediction

    assert(transform.isInstanceOf[DataFrame])
    assert(transform.columns.contains("prediction"))
    assert(transform.columns.contains("rawPrediction"))
  }

  @After def afterAll() {
    spark.stop()
  }
}
