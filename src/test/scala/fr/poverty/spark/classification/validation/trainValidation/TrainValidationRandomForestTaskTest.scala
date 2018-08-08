package fr.poverty.spark.classification.validation.trainValidation

import org.apache.log4j.{Level, LogManager}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.sql.SparkSession
import org.junit.{After, Before, Test}
import org.scalatest.junit.AssertionsForJUnit

/**
  * Created by mahjoubi on 12/06/18.
  */
class TrainValidationRandomForestTaskTest extends AssertionsForJUnit {

  private val labelColumn: String = "target"
  private val featureColumn: String = "features"
  private val predictionColumn: String = "prediction"
  private val ratio: Double = 0.5
  private val pathSave = "target/model/trainValidation/randomForest"
  private var spark: SparkSession = _

  @Before def beforeAll() {
    spark = SparkSession.builder.master("local").appName("train validation random forest task test").getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)
  }

  @Test def testEstimator(): Unit = {
    val randomForest = new TrainValidationRandomForestTask(labelColumn, featureColumn, predictionColumn, pathSave, ratio)
    randomForest.defineEstimator()
    randomForest.defineGridParameters()
    randomForest.defineEvaluator()
    randomForest.defineValidatorModel()

    val estimator = randomForest.getEstimator
    assert(estimator.isInstanceOf[RandomForestClassifier])
    assert(estimator.getLabelCol == labelColumn)
    assert(estimator.getFeaturesCol == featureColumn)
    assert(estimator.getPredictionCol == predictionColumn)



//    val randomForest = new TrainValidationRandomForestTask(labelColumn, featureColumn, predictionColumn, ratio, pathSave)
//    randomForest.defineEstimator()
//    randomForest.defineGridParameters()
//
//    val gridParams = randomForest.getParamGrid
//    assert(gridParams.isInstanceOf[Array[ParamMap]])
//    assert(gridParams.length == 16)



//    val randomForest = new TrainValidationRandomForestTask(labelColumn, featureColumn, predictionColumn, ratio, pathSave)


//    val trainValidator = randomForest.getTrainValidator
//    assert(trainValidator.getEstimator.isInstanceOf[Estimator[_]])
//    assert(trainValidator.getEvaluator.isInstanceOf[MulticlassClassificationEvaluator])
//    assert(trainValidator.getEstimatorParamMaps.isInstanceOf[Array[ParamMap]])
//    assert(trainValidator.getTrainRatio.isInstanceOf[Double])
//    assert(trainValidator.getTrainRatio == ratio)



//    val data = new LoadDataSetTask("src/test/resources", format = "parquet").run(spark, "classificationTask")
////    val randomForest = new TrainValidationRandomForestTask(labelColumn, featureColumn, predictionColumn, ratio, pathSave)
//    randomForest.run(data)
//
//    val model = TrainValidationSplitModel.load(s"$pathSave/model")
//    assert(model.isInstanceOf[TrainValidationSplitModel])
  }

  @After def afterAll() {
    spark.stop()
  }
}