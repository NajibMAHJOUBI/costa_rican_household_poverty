package fr.poverty.spark.classification.trainValidation

import fr.poverty.spark.utils.LoadDataSetTask
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit

/**
  * Created by mahjoubi on 12/06/18.
  */
class TrainValidationDecisionTreeTaskTest extends AssertionsForJUnit {

  private var spark: SparkSession = _

  @Before def beforeAll() {
    spark = SparkSession.builder.master("local").appName("gbt classifier test").getOrCreate()
    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)
  }

  @Test def testDecisionTreeClassifier(): Unit = {
    val data = new LoadDataSetTask("src/test/resources", format = "parquet")
      .run(spark, "classificationTask")
    val decisionTree = new TrainValidationDecisionTreeTask("target",
      "features", "prediction", 0.75, "/target/model/trainValidation/decisionTree")
    decisionTree.run(data)

  }

  @After def afterAll() {
    spark.stop()
  }
}
