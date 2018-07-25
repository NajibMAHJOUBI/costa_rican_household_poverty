package fr.poverty.spark.classification.task

import fr.poverty.spark.utils.LoadDataSetTask
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit

/**
  * Created by mahjoubi on 12/06/18.
  */
class LogisticRegressionTaskTest extends AssertionsForJUnit  {

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

  @Test def testLogisticRegression(): Unit = {
    val data = new LoadDataSetTask("src/test/resources", format = "parquet").run(spark, "classificationTask")
    val logisticRegression = new LogisticRegressionTask(labelColumn = "target",
                                                        featureColumn = "features",
                                                        predictionColumn = "prediction")
    logisticRegression.defineModel
    logisticRegression.fit(data)
    logisticRegression.transform(data)
    val prediction = logisticRegression.getTransform

    assert(prediction.isInstanceOf[DataFrame])
    assert(prediction.columns.contains("prediction"))
    assert(prediction.columns.contains("probability"))
    assert(prediction.columns.contains("rawPrediction"))
  }

  @Test def testRegParam(): Unit = {
    val regParam = 0.5
    val logisticRegression = new LogisticRegressionTask()
    logisticRegression.defineModel
    logisticRegression.setRegParam(regParam)
    assert(logisticRegression.getRegParam == regParam)
  }

  @After def afterAll() {
    spark.stop()
  }
}
