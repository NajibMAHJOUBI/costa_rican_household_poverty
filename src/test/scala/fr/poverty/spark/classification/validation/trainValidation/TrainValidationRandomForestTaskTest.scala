package fr.poverty.spark.classification.validation.trainValidation

import fr.poverty.spark.classification.gridParameters.GridParametersRandomForest
import fr.poverty.spark.utils.LoadDataSetTask
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.TrainValidationSplitModel
import org.apache.spark.sql.SparkSession
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit

/**
  * Created by mahjoubi on 12/06/18.
  */
class TrainValidationRandomForestTaskTest extends AssertionsForJUnit {

  private val labelColumn: String = "target"
  private val featureColumn: String = "features"
  private val predictionColumn: String = "prediction"
  private val metricName: String = "accuracy"
  private val ratio: Double = 0.5
  private val pathSave = "target/model/trainValidation/randomForest"
  private var spark: SparkSession = _

  @Before def beforeAll() {
    spark = SparkSession.builder.master("local").appName("train validation random forest task test").getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)
  }

  @Test def testEstimator(): Unit = {
    val data = new LoadDataSetTask("src/test/resources", format = "parquet").run(spark, "classificationTask")
    val randomForest = new TrainValidationRandomForestTask(labelColumn, featureColumn, predictionColumn,
      metricName, pathSave, ratio)
    randomForest.run(data)

    val estimator = randomForest.getEstimator
    assert(estimator.isInstanceOf[RandomForestClassifier])
    assert(estimator.getLabelCol == labelColumn)
    assert(estimator.getFeaturesCol == featureColumn)
    assert(estimator.getPredictionCol == predictionColumn)

    val gridParams = randomForest.getGridParameters
    assert(gridParams.isInstanceOf[Array[ParamMap]])
    assert(gridParams.length == GridParametersRandomForest.getMaxBins.length * GridParametersRandomForest.getMaxDepth.length * GridParametersRandomForest.getImpurity.length)

    val trainValidator = randomForest.getTrainValidator
    assert(trainValidator.getEstimator.isInstanceOf[Estimator[_]])
    assert(trainValidator.getEvaluator.isInstanceOf[MulticlassClassificationEvaluator])
    assert(trainValidator.getEstimatorParamMaps.isInstanceOf[Array[ParamMap]])
    assert(trainValidator.getTrainRatio.isInstanceOf[Double])
    assert(trainValidator.getTrainRatio == ratio)

    val model = randomForest.getTrainValidatorModel
    assert(model.isInstanceOf[TrainValidationSplitModel])
  }

  @After def afterAll() {
    spark.stop()
  }
}
