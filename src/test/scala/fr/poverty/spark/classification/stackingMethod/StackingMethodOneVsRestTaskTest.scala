package fr.poverty.spark.classification.stackingMethod

import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}


class StackingMethodOneVsRestTaskTest {

  private val pathTrain = "src/test/resources"
  private val pathPrediction = "src/test/resources/stackingTask"
  private val stringIndexerModel = "src/test/resources/stringIndexerModel"
  private val pathSave: String = "target/stackingMethod/oneVsRest"
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
      .appName("test stacking method test")
      .getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)

    listPathPrediction = List("decisionTree", "logisticRegression", "randomForest").map(method => s"$pathPrediction/$method")

  }

  @Test def testStackingOneVsRestCrossValidation(): Unit = {
    val stackingMethodOneVsRest = new StackingMethodOneVsRestTask(
      idColumn = idColumn, labelColumn = labelColumn, predictionColumn = predictionColumn,
      pathPrediction = listPathPrediction, mapFormat,
      pathTrain = pathTrain, formatTrain="csv",
      pathStringIndexer = stringIndexerModel, pathSave = s"$pathSave/crossValidation",
      validationMethod = "crossValidation",
      ratio = 2.0, classifier = "logisticRegression", bernoulliOption = false)
    stackingMethodOneVsRest.run(spark)
    stackingMethodOneVsRest.transform()
    val prediction = stackingMethodOneVsRest.getTransformPrediction
    val submission = stackingMethodOneVsRest.getTransformSubmission

    assert(prediction.isInstanceOf[DataFrame])
    assert(prediction.columns.contains("prediction"))
    assert(submission.isInstanceOf[DataFrame])
    assert(submission.columns.contains("prediction"))
  }

  @Test def testStackingOneVsRestTrainValidation(): Unit = {
    val stackingMethodOneVsRest = new StackingMethodOneVsRestTask(
      idColumn = idColumn, labelColumn = labelColumn, predictionColumn = predictionColumn,
      pathPrediction = listPathPrediction, mapFormat,
      pathTrain = pathTrain, formatTrain="csv",
      pathStringIndexer = stringIndexerModel, pathSave = s"$pathSave/trainValidation",
      validationMethod = "crossValidation",
      ratio = 2.0, classifier = "logisticRegression", bernoulliOption = false)
    stackingMethodOneVsRest.run(spark)
    stackingMethodOneVsRest.transform()
    val prediction = stackingMethodOneVsRest.getTransformPrediction
    val submission = stackingMethodOneVsRest.getTransformSubmission

    assert(prediction.isInstanceOf[DataFrame])
    assert(prediction.columns.contains("prediction"))
    assert(submission.isInstanceOf[DataFrame])
    assert(submission.columns.contains("prediction"))
  }

  @After def afterAll() {
    spark.stop()
  }
}
