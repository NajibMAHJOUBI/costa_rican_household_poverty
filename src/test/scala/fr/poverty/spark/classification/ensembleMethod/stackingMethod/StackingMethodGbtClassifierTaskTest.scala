package fr.poverty.spark.classification.ensembleMethod.stackingMethod

import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}


class StackingMethodGbtClassifierTaskTest {

  private val pathTrain = "src/test/resources"
  private val pathPrediction = "src/test/resources/stackingTask"
  private val stringIndexerModel = "src/test/resources/stringIndexerModel"
  private val pathSave: String = "target/stackingMethod/gbtClassifier"
  private val idColumn = "id"
  private val labelColumn = "target"
  private val predictionColumn = "target"
  private val metricName: String = "accuracy"
  private val mapFormat: Map[String, String] = Map("prediction" -> "parquet", "submission" -> "csv")
  private var listPathPrediction: List[String] = _
  private var spark: SparkSession = _
  private val bernoulliOption: Boolean = false

  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("test stacking method test - GBT classifier")
      .getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)

    listPathPrediction = List("decisionTree", "logisticRegression", "randomForest").map(method => s"$pathPrediction/$method")

  }

  @Test def testStackingGbtClassifierCrossValidation(): Unit = {
    val stackingMethodGbtClassifier = new StackingMethodGbtClassifierTask(
      idColumn = idColumn, labelColumn = labelColumn, predictionColumn = predictionColumn,
      pathPrediction = listPathPrediction, mapFormat,
      pathTrain = pathTrain, formatTrain="csv",
      pathStringIndexer = stringIndexerModel, pathSave = s"$pathSave/crossValidation",
      validationMethod = "crossValidation",
      ratio = 2.0, metricName, bernoulliOption)
    stackingMethodGbtClassifier.run(spark)
    stackingMethodGbtClassifier.transform()
    val prediction = stackingMethodGbtClassifier.getTransformPrediction
    val submission = stackingMethodGbtClassifier.getTransformSubmission

    assert(prediction.isInstanceOf[DataFrame])
    assert(prediction.columns.contains("prediction"))
    assert(submission.isInstanceOf[DataFrame])
    assert(submission.columns.contains("prediction"))
  }

  @Test def testStackingGbtClassifierTrainValidation(): Unit = {
    val stackingMethodGbtClassifier = new StackingMethodGbtClassifierTask(
      idColumn = idColumn, labelColumn = labelColumn, predictionColumn = predictionColumn,
      pathPrediction = listPathPrediction, mapFormat,
      pathTrain = pathTrain, formatTrain="csv",
      pathStringIndexer = stringIndexerModel, pathSave = s"$pathSave/trainValidation",
      validationMethod = "crossValidation",
      ratio = 2.0, metricName, bernoulliOption)
    stackingMethodGbtClassifier.run(spark)
    stackingMethodGbtClassifier.transform()
    val prediction = stackingMethodGbtClassifier.getTransformPrediction
    val submission = stackingMethodGbtClassifier.getTransformSubmission

    assert(prediction.isInstanceOf[DataFrame])
    assert(prediction.columns.contains("prediction"))
    assert(submission.isInstanceOf[DataFrame])
    assert(submission.columns.contains("prediction"))
  }

  @After def afterAll() {
    spark.stop()
  }
}
