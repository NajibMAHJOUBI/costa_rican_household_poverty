package fr.poverty.spark.classification.gridParameters

import org.apache.spark.ml.classification.{DecisionTreeClassifier, LinearSVC, LogisticRegression}

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
}
