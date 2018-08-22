package fr.poverty.spark.classification.validation

import fr.poverty.spark.classification.validation.trainValidation.TrainValidationTask
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.SparkSession
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit

/**
  * Created by mahjoubi on 12/06/18.
  */

class ValidationTaskTest extends AssertionsForJUnit {

  private var spark: SparkSession = _
  private val labelColumn: String = "target"
  private val featureColumn: String = "features"
  private val predictionColumn: String = "prediction"
  private val ratio: Double = 0.5
  private val pathSave = "target/model/trainValidation/decisionTree"

  @Before def beforeAll() {
    spark = SparkSession.builder.master("local").appName("train validation decision tree task test").getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)
  }

  @Test def testEvaluatorAccuracy(): Unit = {
    val trainValidation = new TrainValidationTask(labelColumn, featureColumn, predictionColumn, "accuracy",
      pathSave, ratio)
    trainValidation.defineEvaluator()

    val evaluator = trainValidation.getEvaluator
    assert(evaluator.isInstanceOf[MulticlassClassificationEvaluator])
    assert(evaluator.getLabelCol == labelColumn)
    assert(evaluator.getPredictionCol == predictionColumn)
    assert(evaluator.getMetricName == "accuracy")
  }

  @Test def testEvaluatorF1(): Unit = {
    val trainValidation = new TrainValidationTask(labelColumn, featureColumn, predictionColumn, "f1",
      pathSave, ratio)
    trainValidation.defineEvaluator()

    val evaluator = trainValidation.getEvaluator
    assert(evaluator.isInstanceOf[MulticlassClassificationEvaluator])
    assert(evaluator.getLabelCol == labelColumn)
    assert(evaluator.getPredictionCol == predictionColumn)
    assert(evaluator.getMetricName == "f1")
  }

  @After def afterAll() {
    spark.stop()
  }
}
