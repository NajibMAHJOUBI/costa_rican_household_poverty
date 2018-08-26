package fr.poverty.spark.classification.validation.trainValidation

import fr.poverty.spark.classification.gridParameters.GridParametersDecisionTree
import fr.poverty.spark.utils.LoadDataSetTask
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.TrainValidationSplitModel
import org.apache.spark.sql.SparkSession
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit

/**
  * Created by mahjoubi on 12/06/18.
  *
  * Train validation decision tree classifier - test suite
  *
  */

class TrainValidationDecisionTreeTaskTest extends AssertionsForJUnit {

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

  @Test def testTrainValidationDecisionTreeClassifier(): Unit = {
    val data = new LoadDataSetTask("src/test/resources", format = "parquet").run(spark, "classificationTask")
    val decisionTree = new TrainValidationDecisionTreeTask(labelColumn, featureColumn, predictionColumn,
      metricName, pathSave, ratio)
    decisionTree.run(data)

    val estimator = decisionTree.getEstimator.asInstanceOf[DecisionTreeClassifier]
    assert(estimator.isInstanceOf[DecisionTreeClassifier])
    assert(estimator.getLabelCol == labelColumn)
    assert(estimator.getFeaturesCol == featureColumn)
    assert(estimator.getPredictionCol == predictionColumn)

    val gridParams = decisionTree.getGridParameters
    assert(gridParams.isInstanceOf[Array[ParamMap]])
    assert(gridParams.length == GridParametersDecisionTree.getMaxBins.length * GridParametersDecisionTree.getMaxDepth.length * GridParametersDecisionTree.getImpurity.length)

    val trainValidator = decisionTree.getTrainValidator
    assert(trainValidator.getEstimator.isInstanceOf[Estimator[_]])
    assert(trainValidator.getEvaluator.isInstanceOf[MulticlassClassificationEvaluator])
    assert(trainValidator.getEstimatorParamMaps.isInstanceOf[Array[ParamMap]])
    assert(trainValidator.getTrainRatio.isInstanceOf[Double])
    assert(trainValidator.getTrainRatio == ratio)

    val model = decisionTree.getTrainValidatorModel
    assert(model.isInstanceOf[TrainValidationSplitModel])
  }

  @After def afterAll() {
    spark.stop()
  }
}
