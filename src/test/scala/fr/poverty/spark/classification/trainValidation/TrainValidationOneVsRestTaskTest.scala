package fr.poverty.spark.classification.trainValidation

import fr.poverty.spark.utils.LoadDataSetTask
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.TrainValidationSplitModel
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit

/**
  * Created by mahjoubi on 12/06/18.
  */

class TrainValidationOneVsRestTaskTest extends AssertionsForJUnit {

  private var spark: SparkSession = _
  private var data: DataFrame = _
  private val labelColumn: String = "target"
  private val featureColumn: String = "features"
  private val predictionColumn: String = "prediction"
  private val ratio: Double = 0.5
  private val pathSave = "target/validation/trainValidation/oneVsRest"

  @Before def beforeAll() {
    spark = SparkSession.builder.master("local").appName("train validation decision tree task test").getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)
  }

  @Test def testTrainValidationOneVsRestDecisionTree(): Unit = {
    val data = new LoadDataSetTask("src/test/resources", format = "parquet").run(spark, "classificationTask")
    val oneVsRest = new TrainValidationOneVsRestTask(labelColumn, featureColumn, predictionColumn, ratio,
      s"$pathSave/decisionTree", "decisionTree")
    oneVsRest.run(data)

    val model = TrainValidationSplitModel.load(s"$pathSave/decisionTree/model")
    assert(model.isInstanceOf[TrainValidationSplitModel])
  }

  @Test def testTrainValidationOneVsRestGbtClassifier(): Unit = {
    val data = new LoadDataSetTask("src/test/resources", format = "parquet").run(spark, "classificationTask")
    val oneVsRest = new TrainValidationOneVsRestTask(labelColumn, featureColumn, predictionColumn, ratio,
      s"$pathSave/gbtClassifier", "gbtClassifier")
    oneVsRest.run(data)

    val model = TrainValidationSplitModel.load(s"$pathSave/gbtClassifier/model")
    assert(model.isInstanceOf[TrainValidationSplitModel])
  }

  @Test def testTrainValidationOneVsRestLogisticRegression(): Unit = {
    data = new LoadDataSetTask("src/test/resources", format = "parquet").run(spark, "classificationTask")
    val oneVsRest = new TrainValidationOneVsRestTask(labelColumn, featureColumn, predictionColumn, ratio,
      s"$pathSave/logisticRegression", "logisticRegression")
    oneVsRest.run(data)

    val model = TrainValidationSplitModel.load(s"$pathSave/logisticRegression/model")
    assert(model.isInstanceOf[TrainValidationSplitModel])
  }

  @Test def testTrainValidationOneVsRestNaiveBayes(): Unit = {
    val data = new LoadDataSetTask("src/test/resources", format = "parquet").run(spark, "classificationTask")
    val oneVsRest = new TrainValidationOneVsRestTask(labelColumn, featureColumn, predictionColumn, ratio, s"$pathSave/naiveBayes", "naiveBayes")
    oneVsRest.run(data)

    val model = TrainValidationSplitModel.load(s"$pathSave/naiveBayes/model")
    assert(model.isInstanceOf[TrainValidationSplitModel])
  }

  @Test def testTrainValidationOneVsRestLinearSvc(): Unit = {
    val data = new LoadDataSetTask("src/test/resources", format = "parquet").run(spark, "classificationTask")
    val oneVsRest = new TrainValidationOneVsRestTask(labelColumn, featureColumn, predictionColumn, ratio, s"$pathSave/linearSvc", "linearSvc")
    oneVsRest.run(data)

    val model = TrainValidationSplitModel.load(s"$pathSave/linearSvc/model")
    assert(model.isInstanceOf[TrainValidationSplitModel])
  }

  @Test def testTrainValidationOneVsRestRandomForest(): Unit = {
    val data = new LoadDataSetTask("src/test/resources", format = "parquet").run(spark, "classificationTask")
    val oneVsRest = new TrainValidationOneVsRestTask(labelColumn, featureColumn, predictionColumn, ratio, s"$pathSave/randomForest", "randomForest")
    oneVsRest.run(data)

    val model = TrainValidationSplitModel.load(s"$pathSave/randomForest/model")
    assert(model.isInstanceOf[TrainValidationSplitModel])
  }

  @After def afterAll() {
    spark.stop()
  }
}
