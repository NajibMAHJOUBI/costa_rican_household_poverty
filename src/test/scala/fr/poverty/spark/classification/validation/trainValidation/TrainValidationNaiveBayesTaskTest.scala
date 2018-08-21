package fr.poverty.spark.classification.validation.trainValidation

import fr.poverty.spark.utils.LoadDataSetTask
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.TrainValidationSplitModel
import org.apache.spark.sql.SparkSession
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit

/**
  * Created by mahjoubi on 12/06/18.
  *
  * train validation - naive bayes classifier - test suite
  *
  */

class TrainValidationNaiveBayesTaskTest extends AssertionsForJUnit {

  private var spark: SparkSession = _
  private val labelColumn: String = "target"
  private val featureColumn: String = "features"
  private val predictionColumn: String = "prediction"
  private val metricName: String = "accuracy"
  private val ratio: Double = 0.5
  private val pathSave = "target/model/trainValidation/decisionTree"

  @Before def beforeAll() {
    spark = SparkSession.builder.master("local").appName("train validation decision tree task test").getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)
  }

  @Test def testTrainValidationNaiveBayesClassifier(): Unit = {
    val data = new LoadDataSetTask("src/test/resources", format = "parquet").run(spark, "classificationTask")
    val naiveBayes = new TrainValidationNaiveBayesTask(labelColumn, featureColumn, predictionColumn,
      metricName, pathSave, ratio, false)
    naiveBayes.run(data)

    val estimator = naiveBayes.getEstimator
    assert(estimator.isInstanceOf[NaiveBayes])
    assert(estimator.getLabelCol == labelColumn)
    assert(estimator.getFeaturesCol == featureColumn)
    assert(estimator.getPredictionCol == predictionColumn)

    val gridParams = naiveBayes.getGridParameters
    assert(gridParams.isInstanceOf[Array[ParamMap]])
    assert(gridParams.length == 1)

    val trainValidator = naiveBayes.getTrainValidator
    assert(trainValidator.getEstimator.isInstanceOf[Estimator[_]])
    assert(trainValidator.getEvaluator.isInstanceOf[MulticlassClassificationEvaluator])
    assert(trainValidator.getEstimatorParamMaps.isInstanceOf[Array[ParamMap]])
    assert(trainValidator.getTrainRatio.isInstanceOf[Double])
    assert(trainValidator.getTrainRatio == ratio)

    assert(naiveBayes.getTrainValidatorModel.isInstanceOf[TrainValidationSplitModel])
  }

  @After def afterAll() {
    spark.stop()
  }
}
