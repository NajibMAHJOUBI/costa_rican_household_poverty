package fr.poverty.spark.classification.stackingMethod

import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}


class StackingMethodLogisticRegressionTaskTest {

  private val pathTrain = "src/test/resources"
  private val pathPrediction = "src/test/resources/stackingTask"
  private val idColumn = "id"
  private val labelColumn = "target"
  private val predictionColumn = "target"
  private var listPathPrediction: List[String] = _
  private var spark: SparkSession = _

  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("test stacking method test")
      .getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)

    listPathPrediction = List("decisionTree", "logisticRegression", "randomForest").map(method => s"$pathPrediction/$method")

  }

  @Test def testStackingLogisticRegressionCrossValidation(): Unit = {
    val stackingMethodLogisticRegression = new StackingMethodLogisticRegressionTask(
      pathPrediction = listPathPrediction, formatPrediction="parquet",
      pathTrain = pathTrain, formatTrain="csv",
      pathSave = "",
      validationMethod = "crossValidation",
      ratio = 2.0,
      idColumn = idColumn, labelColumn = labelColumn, predictionColumn = predictionColumn)
    stackingMethodLogisticRegression.run(spark)
    val transform = stackingMethodLogisticRegression.transform(stackingMethodLogisticRegression.getLabelFeatures)
    assert(transform.isInstanceOf[DataFrame])
    assert(transform.columns.contains("prediction"))
  }

  @Test def testStackingLogisticRegressionTrainValidation(): Unit = {
    val stackingMethodLogisticRegression = new StackingMethodLogisticRegressionTask(
      pathPrediction = listPathPrediction, formatPrediction="parquet",
      pathTrain = pathTrain, formatTrain="csv",
      pathSave = "",
      validationMethod = "trainValidation",
      ratio = 0.75,
      idColumn = idColumn, labelColumn = labelColumn, predictionColumn = predictionColumn)
    stackingMethodLogisticRegression.run(spark)
    val transform = stackingMethodLogisticRegression.transform(stackingMethodLogisticRegression.getLabelFeatures)
    assert(transform.isInstanceOf[DataFrame])
    assert(transform.columns.contains("prediction"))
  }

  @After def afterAll() {
    spark.stop()
  }
}
