package fr.poverty.spark.classification.gridParameters

import org.apache.spark.ml.classification._
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.ParamGridBuilder

object GridParametersOneVsRest {

  def defineLogisticRegressionGrid(labelColumn: String, featureColumn: String, predictionColumn: String): Array[LogisticRegression] = {
    val regParam = GridParametersLogisticRegression.getRegParam
    val elasticNetParam = GridParametersLogisticRegression.getElasticNetParam
    val model = new LogisticRegression()
      .setFeaturesCol(featureColumn)
      .setLabelCol(labelColumn)
      .setPredictionCol(predictionColumn)
    val params: Array[(Double, Double)] = for(reg <- regParam; elastic <- elasticNetParam) yield(reg, elastic)
    var paramGrid: Array[LogisticRegression] = Array()
    params.foreach(param => {
                model.setRegParam(param._1).setElasticNetParam(param._2)
                paramGrid = paramGrid ++ Array(model)
    })
    paramGrid
  }

  def defineLinearSvcGrid(labelColumn: String, featureColumn: String, predictionColumn: String): Array[LinearSVC] = {
    val regParam = GridParametersLinearSvc.getRegParam
    val fitIntercept = GridParametersLinearSvc.getFitIntercept
    val standardization = GridParametersLinearSvc.getStandardization

    val model = new LinearSVC()
      .setFeaturesCol(featureColumn)
      .setLabelCol(labelColumn)
      .setPredictionCol(predictionColumn)
    val params: Array[(Double, Boolean, Boolean)] = for(reg <- regParam; fit <- fitIntercept; standard <- standardization) yield(reg, fit, standard)
    var paramGrid: Array[LinearSVC] = Array()
    params.foreach(param => {
      model.setRegParam(param._1).setFitIntercept(param._2).setStandardization(param._3)
      paramGrid = paramGrid ++ Array(model)
    })
    paramGrid
  }

  def defineDecisionTreeGrid(labelColumn: String, featureColumn: String, predictionColumn: String): Array[DecisionTreeClassifier] = {
    val maxDepth = GridParametersDecisionTree.getMaxDepth
    val maxBins = GridParametersDecisionTree.getMaxBins

    val model = new DecisionTreeClassifier()
      .setFeaturesCol(featureColumn)
      .setLabelCol(labelColumn)
      .setPredictionCol(predictionColumn)
    val params: Array[(Int, Int)] = for(depth <- maxDepth; bins <- maxBins) yield(depth, bins)
    var paramGrid: Array[DecisionTreeClassifier] = Array()
    params.foreach(param => {
      model.setMaxDepth(param._1).setMaxBins(param._2)
      paramGrid = paramGrid ++ Array(model)
    })
    paramGrid
  }

  def defineRandomForestGrid(labelColumn: String, featureColumn: String, predictionColumn: String): Array[RandomForestClassifier] = {
    val maxDepth = GridParametersRandomForest.getMaxDepth
    val maxBins = GridParametersRandomForest.getMaxBins

    val model = new RandomForestClassifier()
      .setFeaturesCol(featureColumn)
      .setLabelCol(labelColumn)
      .setPredictionCol(predictionColumn)
    val params: Array[(Int, Int)] = for(depth <- maxDepth; bins <- maxBins) yield(depth, bins)
    var paramGrid: Array[RandomForestClassifier] = Array()
    params.foreach(param => {
      model.setMaxDepth(param._1).setMaxBins(param._2)
      paramGrid = paramGrid ++ Array(model)
    })
    paramGrid
  }

  def defineNaiveBayes(labelColumn: String, featureColumn: String, predictionColumn: String, bernoulliOption: Boolean): Array[NaiveBayes] = {
    val modelType = GridParametersNaiveBayes.getModelType(bernoulliOption)

    val model = new NaiveBayes()
      .setFeaturesCol(featureColumn)
      .setLabelCol(labelColumn)
      .setPredictionCol(predictionColumn)
    val params: Array[String] = for(bernoulli <- modelType) yield(bernoulli)
    var paramGrid: Array[NaiveBayes] = Array()
    params.foreach(param => {
      model.setModelType(param)
      paramGrid = paramGrid ++ Array(model)
    })
    paramGrid
  }

  def defineGbtClassifier(labelColumn: String, featureColumn: String, predictionColumn: String): Array[GBTClassifier] = {
    val maxDepth = GridParametersGbtClassifier.getMaxDepth
    val maxBins = GridParametersGbtClassifier.getMaxBins

    val model = new GBTClassifier()
      .setFeaturesCol(featureColumn)
      .setLabelCol(labelColumn)
      .setPredictionCol(predictionColumn)
    val params: Array[(Int, Int)] = for(depth <- maxDepth; bins <- maxBins) yield(depth, bins)
    var paramGrid: Array[GBTClassifier] = Array()
    params.foreach(param => {
      model.setMaxDepth(param._1).setMaxBins(param._2)
      paramGrid = paramGrid ++ Array(model)
    })
    paramGrid
  }

  def getParamsGrid(estimator: OneVsRest, classifierOption: String, labelColumn: String, featureColumn: String, predictionColumn: String, bernoulliOption: Boolean = false): Array[ParamMap] = {
    val paramGrid: ParamGridBuilder = new ParamGridBuilder()
    if (classifierOption == "logisticRegression"){paramGrid.addGrid(estimator.classifier, defineLogisticRegressionGrid(labelColumn, featureColumn, predictionColumn))}
    else if (classifierOption == "decisionTree") {paramGrid.addGrid(estimator.classifier, defineDecisionTreeGrid(labelColumn, featureColumn, predictionColumn))}
    else if (classifierOption == "linearSvc") {paramGrid.addGrid(estimator.classifier, defineLinearSvcGrid(labelColumn, featureColumn, predictionColumn))}
    else if (classifierOption == "randomForest") {paramGrid.addGrid(estimator.classifier, defineRandomForestGrid(labelColumn, featureColumn, predictionColumn))}
    else if(classifierOption == "naiveBayes") {paramGrid.addGrid(estimator.classifier, defineNaiveBayes(labelColumn, featureColumn, predictionColumn, bernoulliOption))}
    else if (classifierOption == "gbtClassifier") {paramGrid.addGrid(estimator.classifier, defineGbtClassifier(labelColumn, featureColumn, predictionColumn))}
    paramGrid.build()
  }

}
