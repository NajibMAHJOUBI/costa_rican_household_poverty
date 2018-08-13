package fr.poverty.spark.classification.stackingMethod

import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}


class StackingMethodRandomForestTaskTest {

  private val pathTrain = "src/test/resources"
  private val pathPrediction = "src/test/resources/stackingTask"
  private val stringIndexerModel = "src/test/resources/stringIndexerModel"
  private val pathSave: String = "target/stackingMethod/randomForest"
  private val idColumn = "id"
  private val labelColumn = "target"
  private val predictionColumn = "target"
  private val mapFormat: Map[String, String] = Map("prediction" -> "parquet", "submission" -> "csv")
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
      pathPrediction = listPathPrediction, mapFormat,
      pathTrain = pathTrain, formatTrain="csv",
      pathStringIndexer = stringIndexerModel, pathSave = s"$pathSave/crossValidation",
      validationMethod = "crossValidation",
      ratio = 2.0)
    stackingMethodRandomForest.run(spark)
    stackingMethodRandomForest.transform()
    val prediction = stackingMethodRandomForest.getTransformPrediction
    val submission = stackingMethodRandomForest.getTransformSubmission

    assert(prediction.isInstanceOf[DataFrame])
    assert(prediction.columns.contains("prediction"))
    assert(submission.isInstanceOf[DataFrame])
    assert(submission.columns.contains("prediction"))
  }

  @Test def testStackingRandomForestTrainValidation(): Unit = {
    val stackingMethodRandomForest = new StackingMethodRandomForestTask(
      idColumn = idColumn, labelColumn = labelColumn, predictionColumn = predictionColumn,
      pathPrediction = listPathPrediction, mapFormat,
      pathTrain = pathTrain, formatTrain="csv",
      pathStringIndexer = stringIndexerModel, pathSave = s"$pathSave/trainValidation",
      validationMethod = "trainValidation",
      ratio = 0.75)
    stackingMethodRandomForest.run(spark)
    stackingMethodRandomForest.transform()
    val prediction = stackingMethodRandomForest.getTransformPrediction
    val submission = stackingMethodRandomForest.getTransformSubmission

    assert(prediction.isInstanceOf[DataFrame])
    assert(prediction.columns.contains("prediction"))
    assert(submission.isInstanceOf[DataFrame])
    assert(submission.columns.contains("prediction"))
  }

  @After def afterAll() {
    spark.stop()
  }
}
