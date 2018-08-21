package fr.poverty.spark.classification.validation.crossValidation

import fr.poverty.spark.utils.LoadDataSetTask
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit

/**
  * Created by mahjoubi on 12/06/18.
  *
  * Cross-validation of decision tree classifier model
  *
  */


class CrossValidationOneVsRestTaskTest extends AssertionsForJUnit {

  private val labelColumn: String = "target"
  private val featureColumn: String = "features"
  private val predictionColumn: String = "prediction"
  private val metricName: String = "accuracy"
  private val numFolds: Integer = 2
  private val pathSave: String = "target/validation/crossValidation/oneVsRest"
  private var spark: SparkSession = _
  private var data: DataFrame = _

  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("test cross validator one vs rest")
      .getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)

    data = new LoadDataSetTask("src/test/resources", "parquet").run(spark, "classificationTask")

  }

  @Test def testCrossValidationOneVsRestDecisionTree(): Unit = {
    val cv = new CrossValidationOneVsRestTask(labelColumn, featureColumn, predictionColumn,
      metricName, s"$pathSave/decisionTree", numFolds, "decisionTree")
    cv.run(data)

    assert(cv.getCrossValidatorModel.isInstanceOf[CrossValidatorModel])
    }

  @Test def testCrossValidationOneVsRestRandomForest(): Unit = {
    val cv = new CrossValidationOneVsRestTask(labelColumn, featureColumn, predictionColumn, metricName,
      s"$pathSave/randomForest", numFolds, "randomForest")
    cv.run(data)

    assert(cv.getCrossValidatorModel.isInstanceOf[CrossValidatorModel])
  }

  @Test def testCrossValidationOneVsRestGbtClassifier(): Unit = {
    val cv = new CrossValidationOneVsRestTask(labelColumn, featureColumn, predictionColumn, metricName,
      s"$pathSave/gbtClassifier", numFolds, "gbtClassifier")
    cv.run(data)

    assert(cv.getCrossValidatorModel.isInstanceOf[CrossValidatorModel])
  }

  @Test def testCrossValidationOneVsRestLogisticRegression(): Unit = {
    val cv = new CrossValidationOneVsRestTask(labelColumn, featureColumn, predictionColumn, metricName,
      s"$pathSave/logisticRegression", numFolds, "logisticRegression")
    cv.run(data)

    assert(cv.getCrossValidatorModel.isInstanceOf[CrossValidatorModel])
  }

  @After def afterAll() {
    spark.stop()
  }

}
