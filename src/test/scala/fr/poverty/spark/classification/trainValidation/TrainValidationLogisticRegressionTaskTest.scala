package fr.poverty.spark.classification.trainValidation

import fr.poverty.spark.utils.LoadDataSetTask
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.TrainValidationSplitModel
import org.apache.spark.sql.SparkSession
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit

/**
  * Created by mahjoubi on 12/06/18.
  */
class TrainValidationLogisticRegressionTaskTest extends AssertionsForJUnit {

  private val labelColumn: String = "target"
  private val featureColumn: String = "features"
  private val predictionColumn: String = "prediction"
  private val ratio: Double = 0.5
  private val pathSave = "target/model/trainValidation/logisticRegression"
  private var spark: SparkSession = _

  @Before def beforeAll() {
    spark = SparkSession.builder.master("local").appName("train validation random forest task test").getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)
  }

  @Test def testEstimator(): Unit = {
    val logisticRegression = new TrainValidationLogisticRegressionTask(labelColumn, featureColumn, predictionColumn, ratio, pathSave)
    logisticRegression.defineEstimator()

    val estimator = logisticRegression.getEstimator
    assert(estimator.isInstanceOf[LogisticRegression])
    assert(estimator.getLabelCol == labelColumn)
    assert(estimator.getFeaturesCol == featureColumn)
    assert(estimator.getPredictionCol == predictionColumn)
  }

  @Test def testGridParameters(): Unit = {
    val logisticRegression = new TrainValidationLogisticRegressionTask(labelColumn, featureColumn, predictionColumn, ratio, pathSave)
    logisticRegression.defineEstimator()
    logisticRegression.defineGridParameters()

    val gridParams = logisticRegression.getParamGrid
    assert(gridParams.isInstanceOf[Array[ParamMap]])
    assert(gridParams.length == 30)
  }

  @Test def testEvaluator(): Unit = {
    val logisticRegression = new TrainValidationLogisticRegressionTask(labelColumn, featureColumn, predictionColumn, ratio, pathSave)
    logisticRegression.defineEstimator()
    logisticRegression.defineGridParameters()
    logisticRegression.defineEvaluator()

    val evaluator = logisticRegression.getEvaluator
    assert(evaluator.isInstanceOf[MulticlassClassificationEvaluator])
    assert(evaluator.getLabelCol == labelColumn)
    assert(evaluator.getPredictionCol == predictionColumn)
    assert(evaluator.getMetricName == "accuracy")
  }

  @Test def testTrainValidator(): Unit = {
    val logisticRegression = new TrainValidationLogisticRegressionTask(labelColumn, featureColumn, predictionColumn, ratio, pathSave)
    logisticRegression.defineEstimator()
    logisticRegression.defineGridParameters()
    logisticRegression.defineEvaluator()
    logisticRegression.defineTrainValidatorModel()

    val trainValidator = logisticRegression.getTrainValidator
    assert(trainValidator.getEstimator.isInstanceOf[Estimator[_]])
    assert(trainValidator.getEvaluator.isInstanceOf[MulticlassClassificationEvaluator])
    assert(trainValidator.getEstimatorParamMaps.isInstanceOf[Array[ParamMap]])
    assert(trainValidator.getTrainRatio.isInstanceOf[Double])
    assert(trainValidator.getTrainRatio == ratio)
  }

  @Test def testTrainValidationLogisticRegressionClassifier(): Unit = {
    val data = new LoadDataSetTask("src/test/resources", format = "parquet").run(spark, "classificationTask")
    val logisticRegression = new TrainValidationLogisticRegressionTask(labelColumn, featureColumn, predictionColumn, ratio, pathSave)
    logisticRegression.run(data)

    val model = TrainValidationSplitModel.load(s"$pathSave/model")
    assert(model.isInstanceOf[TrainValidationSplitModel])
  }

  @After def afterAll() {
    spark.stop()
  }
}