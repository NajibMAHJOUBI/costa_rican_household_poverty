package fr.poverty.spark.classification.trainValidation

import fr.poverty.spark.utils.LoadDataSetTask
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.TrainValidationSplitModel
import org.apache.spark.sql.SparkSession
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit

/**
  * Created by mahjoubi on 12/06/18.
  */

class TrainValidationOneVsRestTaskTest extends AssertionsForJUnit {

  private var spark: SparkSession = _
  private val labelColumn: String = "target"
  private val featureColumn: String = "features"
  private val predictionColumn: String = "prediction"
  private val ratio: Double = 0.5
  private val pathSave = "target/model/trainValidation/oneVsRest"
  private val classifier: String = "logisticRegression"

  @Before def beforeAll() {
    spark = SparkSession.builder.master("local").appName("train validation decision tree task test").getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)
  }

  @Test def testEstimator(): Unit = {
    val oneVsRest = new TrainValidationOneVsRestTask(labelColumn, featureColumn, predictionColumn, ratio, pathSave, classifier)
    oneVsRest.defineEstimator()

    val estimator = oneVsRest.getEstimator
    assert(estimator.isInstanceOf[OneVsRest])
    assert(estimator.getLabelCol == labelColumn)
    assert(estimator.getFeaturesCol == featureColumn)
    assert(estimator.getPredictionCol == predictionColumn)
  }

  @Test def testGridParameters(): Unit = {
    val oneVsRest = new TrainValidationOneVsRestTask(labelColumn, featureColumn, predictionColumn, ratio, pathSave, classifier)
    oneVsRest.defineEstimator()
    oneVsRest.defineGridParameters()

    val gridParams = oneVsRest.getParamGrid
    assert(gridParams.isInstanceOf[Array[ParamMap]])
    assert(gridParams.length == 30)

    gridParams.foreach(param => {println(s"${param}")})
  }

  @Test def testTrainValidator(): Unit = {
    val oneVsRest = new TrainValidationOneVsRestTask(labelColumn, featureColumn, predictionColumn, ratio, pathSave, classifier)
    oneVsRest.defineEstimator()
    oneVsRest.defineGridParameters()
    oneVsRest.defineEvaluator()
    oneVsRest.defineTrainValidatorModel()

    val trainValidator = oneVsRest.getTrainValidator
    assert(trainValidator.getEstimator.isInstanceOf[Estimator[_]])
    assert(trainValidator.getEvaluator.isInstanceOf[MulticlassClassificationEvaluator])
    assert(trainValidator.getEstimatorParamMaps.isInstanceOf[Array[ParamMap]])
    assert(trainValidator.getTrainRatio.isInstanceOf[Double])
    assert(trainValidator.getTrainRatio == ratio)
  }

  @Test def testTrainValidationOneVsRestClassifierLogisticRegression(): Unit = {
    val data = new LoadDataSetTask("src/test/resources", format = "parquet").run(spark, "classificationTask")
    val oneVsRest = new TrainValidationOneVsRestTask(labelColumn, featureColumn, predictionColumn, ratio, pathSave, classifier)
    oneVsRest.run(data)

    val model = TrainValidationSplitModel.load(s"$pathSave/model")
    assert(model.isInstanceOf[TrainValidationSplitModel])
  }

  @Test def testTrainValidationOneVsRestClassifierDecisionTree(): Unit = {
    val data = new LoadDataSetTask("src/test/resources", format = "parquet").run(spark, "classificationTask")
    val oneVsRest = new TrainValidationOneVsRestTask(labelColumn, featureColumn, predictionColumn, ratio, pathSave, "decisionTree")
    oneVsRest.run(data)

    val model = TrainValidationSplitModel.load(s"$pathSave/model")
    assert(model.isInstanceOf[TrainValidationSplitModel])
  }

  @Test def testTrainValidationOneVsRestClassifierLinearSvc(): Unit = {
    val data = new LoadDataSetTask("src/test/resources", format = "parquet").run(spark, "classificationTask")
    val oneVsRest = new TrainValidationOneVsRestTask(labelColumn, featureColumn, predictionColumn, ratio, pathSave, "linearSvc")
    oneVsRest.run(data)

    val model = TrainValidationSplitModel.load(s"$pathSave/model")
    assert(model.isInstanceOf[TrainValidationSplitModel])
  }

  @After def afterAll() {
    spark.stop()
  }
}