package fr.poverty.spark.classification.validation.crossValidation

import fr.poverty.spark.utils.LoadDataSetTask
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.ml.classification.GBTClassifier
import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit

/**
  * Created by mahjoubi on 12/06/18.
  *
  * Cross-validation of GBT classifier model
  *
  */


class CrossValidationGbtClassifierTaskTest extends AssertionsForJUnit {

  private val labelColumn: String = "target"
  private val featureColumn: String = "features"
  private val predictionColumn: String = "prediction"
  private var spark: SparkSession = _

  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("test cross validation - gbt classifier")
      .getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)
  }

  @Test def testCrossValidationGbtClassifier(): Unit = {
    val data = new LoadDataSetTask("src/test/resources", "parquet").run(spark, "classificationTask")

    val cv = new CrossValidationGbtClassifierTask(
      labelColumn = labelColumn,
      featureColumn = featureColumn,
      predictionColumn = predictionColumn,
      numFolds = 2,
      pathSave = "target/validation/crossValidation/gbtClassifier")
    cv.run(data)

    assert(cv.getLabelColumn == labelColumn)
    assert(cv.getFeatureColumn == featureColumn)
    assert(cv.getPredictionColumn == predictionColumn)
    assert(cv.getGridParameters.isInstanceOf[Array[ParamMap]])
    assert(cv.getEstimator.isInstanceOf[GBTClassifier])
    assert(cv.getEvaluator.isInstanceOf[Evaluator])
    assert(cv.getCrossValidator.isInstanceOf[CrossValidator])

    val bestModel = cv.getBestModel
    assert(bestModel.getLabelCol == labelColumn)
    assert(bestModel.getFeaturesCol == featureColumn)
    assert(bestModel.getPredictionCol == predictionColumn)

    cv.transform(data)
    val transform = cv.getPrediction
    assert(transform.isInstanceOf[DataFrame])
    assert(transform.count() == data.count())
    assert(transform.columns.contains(predictionColumn))
    }

  @After def afterAll() {
    spark.stop()
  }

}
