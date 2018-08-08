package fr.poverty.spark.classification.stackingMethod

import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}


class StackingMethodRandomForestTaskTest {

  private val pathTrain = "src/test/resources"
  private val pathPrediction = "src/test/resources/stackingTask"
  private val stringIndexerModel = "src/test/resources/stringIndexerModel"
  private val idColumn = "id"
  private val labelColumn = "target"
  private val predictionColumn = "target"
  private var listPathPrediction: List[String] = _
  private var spark: SparkSession = _

  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("test stacking method test - random forest")
      .getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)

    listPathPrediction = List("decisionTree", "logisticRegression", "randomForest").map(method => s"$pathPrediction/$method")

  }

  @Test def testStackingRandomForestCrossValidation(): Unit = {
    val stackingMethodRandomForest = new StackingMethodRandomForestTask(
      idColumn = idColumn, labelColumn = labelColumn, predictionColumn = predictionColumn,
      pathPrediction = listPathPrediction, formatPrediction="parquet",
      pathTrain = pathTrain, formatTrain="csv",
      pathStringIndexer = stringIndexerModel, pathSave = "",
      validationMethod = "crossValidation",
      ratio = 2.0)
    stackingMethodRandomForest.run(spark)
    val transform = stackingMethodRandomForest.transform(stackingMethodRandomForest.getLabelFeatures)
    transform.show()
    assert(transform.isInstanceOf[DataFrame])
    assert(transform.columns.contains("prediction"))
  }

  @Test def testStackingRandomForestTrainValidation(): Unit = {
    val stackingMethodRandomForest = new StackingMethodRandomForestTask(
      idColumn = idColumn, labelColumn = labelColumn, predictionColumn = predictionColumn,
      pathPrediction = listPathPrediction, formatPrediction="parquet",
      pathTrain = pathTrain, formatTrain="csv",
      pathStringIndexer = stringIndexerModel, pathSave = "",
      validationMethod = "trainValidation",
      ratio = 0.75)
    stackingMethodRandomForest.run(spark)
    val transform = stackingMethodRandomForest.transform(stackingMethodRandomForest.getLabelFeatures)
    transform.show()
    assert(transform.isInstanceOf[DataFrame])
    assert(transform.columns.contains("prediction"))
  }

  @After def afterAll() {
    spark.stop()
  }
}
