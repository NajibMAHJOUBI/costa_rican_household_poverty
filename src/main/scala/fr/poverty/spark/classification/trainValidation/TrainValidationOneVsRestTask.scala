//package fr.poverty.spark.classification.trainValidation
//
//
//import fr.poverty.spark.classification.task.OneVsRestTask
//import org.apache.spark.ml.classification.OneVsRest
//import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
//import org.apache.spark.sql.DataFrame
//
//
//class TrainValidationOneVsRestTask(override val labelColumn: String,
//                                   override val featureColumn: String,
//                                   override val predictionColumn: String,
//                                   override val trainRatio: Double,
//                                   override val pathSave: String,
//                                   val classifier: String)
//  extends TrainValidationTask(labelColumn, featureColumn, predictionColumn, trainRatio, pathSave)
//    with TrainValidationModelFactory {
//
//  var estimator: OneVsRest = _
//
//  override def run(data: DataFrame): TrainValidationOneVsRestTask = {
//    defineEstimator()
//    defineGridParameters()
//    defineEvaluator()
//    defineTrainValidatorModel()
//    fit(data)
//    saveModel()
//    this
//  }
//
//  override def defineEstimator(): TrainValidationOneVsRestTask = {
//    estimator = new OneVsRestTask(labelColumn, featureColumn, predictionColumn, classifier).defineModel.getModel
//    this
//  }
//  override def defineGridParameters(): TrainValidationOneVsRestTask = {
//    paramGrid = new ParamGridBuilder()
//      .addGrid(estimator.regParam, Array(0.0, 0.001, 0.01, 0.1, 1.0, 10.0))
//      .addGrid(estimator.elasticNetParam, Array(0.0, 0.25, 0.5, 0.75, 1.0))
//      .build()
//    this
//  }
//
//  override def defineTrainValidatorModel(): TrainValidationOneVsRestTask = {
//    trainValidator = new TrainValidationSplit()
//      .setEvaluator(evaluator)
//      .setEstimatorParamMaps(paramGrid)
//      .setEstimator(estimator)
//      .setTrainRatio(trainRatio)
//    this
//  }
//
//  def getEstimator: LogisticRegression = {
//    estimator
//  }
//
//  def getBestModel: LogisticRegressionModel = {
//    trainValidatorModel.bestModel.asInstanceOf[LogisticRegressionModel]
//  }
//}