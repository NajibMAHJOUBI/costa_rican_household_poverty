package fr.poverty.spark

import fr.poverty.spark.classification.task.DecisionTreeTask
import fr.poverty.spark.utils.LoadDataSetTask
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit

/**
  * Created by mahjoubi on 12/06/18.
  */


class DecisionTreeTaskTest extends AssertionsForJUnit  {

  private var spark: SparkSession = _

  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("decision tree test")
      .getOrCreate()
    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)
  }

  @Test def testDecisionTreeClassifier(): Unit = {
    val data = new LoadDataSetTask("src/test/resources", format = "parquet").run(spark, "classificationTask")
    val decisionTree = new DecisionTreeTask(labelColumn = "target", featureColumn = "features", predictionColumn = "prediction")
    decisionTree.defineModel
    decisionTree.fit(data)
    decisionTree.transform(data)
    val transform = decisionTree.getPrediction

    assert(transform.isInstanceOf[DataFrame])
    assert(transform.columns.contains("prediction"))
    assert(transform.columns.contains("probability"))
    assert(transform.columns.contains("rawPrediction"))
  }

  @After def afterAll() {
    spark.stop()
  }
}
