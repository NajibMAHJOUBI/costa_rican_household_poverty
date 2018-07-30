package fr.poverty.spark.classification.task

import org.apache.spark.ml.classification.{Classifier, OneVsRest, OneVsRestModel}
import org.apache.spark.sql.DataFrame

class OneVsRestTask(override val labelColumn: String, override val featureColumn: String, override val predictionColumn: String, val classifier: String) extends ClassificationModelTask(labelColumn, featureColumn, predictionColumn) with ClassificationModelFactory {

  var model: OneVsRest = _
  var modelFit: OneVsRestModel = _
  var classifierModel: Classifier[_, _, _] = _

  override def defineModel: OneVsRestTask = {
    getBaseClassifier
    model = new OneVsRest().setFeaturesCol(featureColumn).setLabelCol(labelColumn).setPredictionCol(predictionColumn).setClassifier(classifierModel)
    this
  }

  def getBaseClassifier: OneVsRestTask = {
    if (classifier == "logisticRegression") {
      classifierModel = new LogisticRegressionTask(labelColumn, featureColumn, predictionColumn).defineModel.getModel
    } else if (classifier == "decisionTree") {
      classifierModel = new DecisionTreeTask(labelColumn, featureColumn, predictionColumn).defineModel.getModel
    } else if (classifier == "linearSvc") {
      classifierModel = new LinearSvcTask(labelColumn, featureColumn, predictionColumn).defineModel.getModel
    } else if (classifier == "randomForest") {
      classifierModel = new RandomForestTask(labelColumn, featureColumn, predictionColumn).defineModel.getModel
    }
    this
  }

  override def fit(data: DataFrame): OneVsRestTask = {
    modelFit = getModel.fit(data)
    this
  }

  def getModel: OneVsRest = {
    model
  }

  override def transform(data: DataFrame): OneVsRestTask = {
    prediction = modelFit.transform(data)
    this
  }

  override def loadModel(path: String): OneVsRestTask = {
    modelFit = OneVsRestModel.load(path)
    this
  }

  override def saveModel(path: String): OneVsRestTask = {
    model.write.overwrite().save(path)
    this
  }

  def getModelFit: OneVsRestModel = {
    modelFit
  }


}
