package fr.poverty.spark.classification.stackingMethod

import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.types.{IntegerType, StringType}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}


class StackingMethodTaskTest {

  private val pathTrain = "src/test/resources"
  private val pathPrediction = "src/test/resources/stackingTask"
  private val idColumn = "id"
  private val labelColumn = "target"
  private val predictionColumn = "target"
  private var stackingMethod: StackingMethodTask = _
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

    stackingMethod = new StackingMethodTask(
      pathPrediction = listPathPrediction, formatPrediction="parquet",
      pathTrain = pathTrain, formatTrain="csv",
      pathSave = "",
      validationMethod = "",
      ratio = 0.0,
      idColumn = idColumn, labelColumn = labelColumn, predictionColumn = predictionColumn)
  }

  @Test def testLoadDataPredictionByLabel(): Unit = {
    val method = "logisticRegression"
    val data = stackingMethod.loadDataPredictionByLabel(spark, s"$pathPrediction/$method", listPathPrediction.indexOf(s"$pathPrediction/$method"))
    assert(data.isInstanceOf[DataFrame])
    assert(data.columns.length == 2)
    assert(data.columns.contains("id"))
    assert(data.columns.contains(s"prediction_${listPathPrediction.indexOf(s"$pathPrediction/$method")}"))
  }

  @Test def testLoadDataLabel(): Unit = {
    val data = stackingMethod.loadDataLabel(spark)
    assert(data.isInstanceOf[DataFrame])
    assert(data.columns.length == 2)
    assert(data.columns.contains("id"))
    assert(data.columns.contains("label"))
  }

  @Test def testMergeData(): Unit = {
    stackingMethod.mergeData(spark)
    val data = stackingMethod.getData
    assert(data.isInstanceOf[DataFrame])
    assert(data.columns.length == listPathPrediction.length + 2)
    assert(data.columns.contains("label"))
    listPathPrediction.foreach(path => assert(data.columns.contains(s"prediction_${listPathPrediction.indexOf(path)}")))
  }

  @Test def testCreateLabelFeatures(): Unit = {
    stackingMethod.mergeData(spark)
    val labelFeatures = stackingMethod.createLabelFeatures(spark)
    assert(labelFeatures.isInstanceOf[DataFrame])
    assert(labelFeatures.columns.length == 3)
    assert(labelFeatures.columns.contains(idColumn))
    assert(labelFeatures.columns.contains(predictionColumn))
    assert(labelFeatures.columns.contains("features"))
    val dataSchema = labelFeatures.schema
    assert(dataSchema.fields(dataSchema.fieldIndex(idColumn)).dataType == StringType)
    assert(dataSchema.fields(dataSchema.fieldIndex(predictionColumn)).dataType == IntegerType)
    assert(dataSchema.fields(dataSchema.fieldIndex("features")).dataType.typeName == "vector")
  }

  @After def afterAll() {
    spark.stop()
  }
}
