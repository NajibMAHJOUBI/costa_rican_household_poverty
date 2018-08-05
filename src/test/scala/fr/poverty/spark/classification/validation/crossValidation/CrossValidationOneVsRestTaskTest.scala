package fr.poverty.spark.classification.validation.crossValidation

import fr.poverty.spark.utils.LoadDataSetTask
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit

/**
  * Created by mahjoubi on 12/06/18.
  *
  * Cross-validation of decision tree classifier model
  *
  */


class CrossValidationOneVsRestTaskTest extends AssertionsForJUnit {

  private val labelColumn: String = "target"
  private val featureColumn: String = "features"
  private val predictionColumn: String = "prediction"
  private val numFolds: Integer = 2
  private val pathSave: String = "target/validation/crossValidation/oneVsRest"
  private var spark: SparkSession = _
  private var data: DataFrame = _

  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("test cross validator one vs rest")
      .getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)

    data = new LoadDataSetTask("src/test/resources", "parquet").run(spark, "classificationTask")
  }

  @Test def testCrossValidationOneVsRestDecisionTree(): Unit = {
    val cv = new CrossValidationOneVsRestTask(
      labelColumn = labelColumn,
      featureColumn = featureColumn,
      predictionColumn = predictionColumn,
      numFolds = numFolds,
      pathSave = s"$pathSave/decisionTree",
      classifier="decisionTree")
    cv.run(data)

    val model = cv.getCrossValidatorModel
    assert(model.isInstanceOf[CrossValidatorModel])
    }

  @Test def testCrossValidationOneVsRestRandomForest(): Unit = {
    val cv = new CrossValidationOneVsRestTask(
      labelColumn = labelColumn,
      featureColumn = featureColumn,
      predictionColumn = predictionColumn,
      numFolds = numFolds,
      pathSave = s"$pathSave/randomForest",
      classifier = "randomForest")
    cv.run(data)

    val model = cv.getCrossValidatorModel
    assert(model.isInstanceOf[CrossValidatorModel])
  }

  @Test def testCrossValidationOneVsRestGbtClassifier(): Unit = {
    val cv = new CrossValidationOneVsRestTask(
      labelColumn = labelColumn,
      featureColumn = featureColumn,
      predictionColumn = predictionColumn,
      numFolds = numFolds,
      pathSave = s"$pathSave/gbtClassifier",
      classifier = "gbtClassifier")
    cv.run(data)

    val model = cv.getCrossValidatorModel
    assert(model.isInstanceOf[CrossValidatorModel])
  }

//  @Test def testCrossValidationOneVsRestNaiveBayes(): Unit = {
//    val cv = new CrossValidationOneVsRestTask(
//      labelColumn = labelColumn,
//      featureColumn = featureColumn,
//      predictionColumn = predictionColumn,
//      numFolds = numFolds,
//      pathSave = s"$pathSave/naiveBayes",
//      classifier = "naiveBayes",
//      bernoulliOption = false)
////    cv.run(data)
//
//    val estimator = cv.defineEstimator().getEstimator
//    val gridParameters = cv.defineGridParameters().getGridParameters
//    val evaluator = cv.defineEvaluator().getEvaluator
//    val crossValidator = cv.defineCrossValidatorModel().getCrossValidator
////    val crossValidatorModel = cv.fit(data).getCrossValidatorModel
//
//    println(s"estimator: $estimator")
//    gridParameters.foreach(println)
//    println(crossValidator.getEstimator)
//    println(crossValidator.getEstimatorParamMaps)
//    println(crossValidator.getEvaluator)
//    println(crossValidator.getNumFolds)
//    println(crossValidator.getClass)
//    println(s"crossValidator: $crossValidator")
//
//
//
//    cv.getCrossValidator.fit(data)
//
//
////    println(s"crossValidatorModel: $crossValidatorModel")
////
////    val model = CrossValidatorModel.load(s"$pathSave/naiveBayes/model")
////    assert(model.isInstanceOf[CrossValidatorModel])
//  }

  @Test def testCrossValidationOneVsRestLogisticRegression(): Unit = {
    val cv = new CrossValidationOneVsRestTask(
      labelColumn = labelColumn,
      featureColumn = featureColumn,
      predictionColumn = predictionColumn,
      numFolds = numFolds,
      pathSave = s"$pathSave/logisticRegression",
      classifier = "logisticRegression")
    cv.run(data)

    val model = CrossValidatorModel.load(s"$pathSave/logisticRegression/model")
    assert(model.isInstanceOf[CrossValidatorModel])
  }

  @After def afterAll() {
    spark.stop()
  }

}
