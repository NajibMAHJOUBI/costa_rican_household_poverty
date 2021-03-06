package fr.poverty.spark.classification.ensembleMethod.stackingMethod

import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.types.{DoubleType, StringType}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}


class StackingMethodTaskTest {

  private val pathTrain = "src/test/resources"
  private val pathPrediction = "src/test/resources/stackingTask"
  private val idColumn = "id"
  private val labelColumn = "target"
  private val predictionColumn = "target"
  private val metricName: String = "accuracy"
  private val listPathPrediction: List[String] = List("decisionTree", "logisticRegression", "randomForest").map(method => s"$pathPrediction/$method")
  private val stackingMethod: StackingMethodTask = new StackingMethodTask(
    idColumn = idColumn, labelColumn = labelColumn, predictionColumn = predictionColumn,
    pathPrediction = listPathPrediction, mapFormat=Map("prediction" -> "parquet", "submission" -> "csv"),
    pathTrain = pathTrain, formatTrain="csv",
    pathStringIndexer = "src/test/resources/stringIndexerModel", pathSave = "",
    validationMethod = "crossValidation",
    ratio =0.0, metricName)
  private var spark: SparkSession = _

  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("test stacking method test")
      .getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)
  }

  @Test def testLoadDataPredictionByLabel(): Unit = {
    stackingMethod.loadStringIndexerModel()
    val method = "logisticRegression"
    val data = stackingMethod.loadDataPredictionByLabel(spark, s"$pathPrediction/$method", listPathPrediction.indexOf(s"$pathPrediction/$method"), "prediction")
    assert(data.isInstanceOf[DataFrame])
    assert(data.columns.length == 2)
    assert(data.columns.contains("id"))
    assert(data.columns.contains(s"prediction_${listPathPrediction.indexOf(s"$pathPrediction/$method")}"))
  }

  @Test def testLoadDataLabel(): Unit = {
    stackingMethod.loadStringIndexerModel()
    val data = stackingMethod.loadDataLabel(spark, "prediction").data
    assert(data.isInstanceOf[DataFrame])
    assert(data.columns.length == 2)
    assert(data.columns.contains("id"))
    assert(data.columns.contains("label"))
  }

  @Test def testMergeData(): Unit = {
    stackingMethod.mergeData(spark, "prediction")
    val data = stackingMethod.data
    assert(data.isInstanceOf[DataFrame])
    assert(data.columns.length == listPathPrediction.length + 2)
    assert(data.columns.contains("id"))
    assert(data.columns.contains("label"))
    listPathPrediction.foreach(path => assert(data.columns.contains(s"prediction_${listPathPrediction.indexOf(path)}")))
  }

  @Test def testCreateLabelFeatures(): Unit = {
    stackingMethod.mergeData(spark, "prediction")
    val labelFeatures = stackingMethod.createLabelFeatures(spark, "prediction")
    assert(labelFeatures.isInstanceOf[DataFrame])
    assert(labelFeatures.columns.length == 3)
    assert(labelFeatures.columns.contains(idColumn))
    assert(labelFeatures.columns.contains(predictionColumn))
    assert(labelFeatures.columns.contains("features"))
    val dataSchema = labelFeatures.schema
    assert(dataSchema.fields(dataSchema.fieldIndex(idColumn)).dataType == StringType)
    assert(dataSchema.fields(dataSchema.fieldIndex(predictionColumn)).dataType == DoubleType)
    assert(dataSchema.fields(dataSchema.fieldIndex("features")).dataType.typeName == "vector")
  }

  @After def afterAll() {
    spark.stop()
  }

}
