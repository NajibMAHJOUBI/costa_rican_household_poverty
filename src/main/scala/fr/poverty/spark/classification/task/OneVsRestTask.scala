package fr.poverty.spark.classification.task

import org.apache.spark.ml.classification.{Classifier, OneVsRest, OneVsRestModel}

class OneVsRestTask(override val labelColumn: String, override val featureColumn: String, override val predictionColumn: String, val classifier: String)
  extends ClassificationModelTask(labelColumn, featureColumn, predictionColumn)
    with ClassificationModelFactory {

  var classifierModel: Classifier[_,_,_] = _

  override def defineEstimator: OneVsRestTask = {
    getBaseClassifier
    estimator = new OneVsRest().setFeaturesCol(featureColumn).setLabelCol(labelColumn).setPredictionCol(predictionColumn).setClassifier(classifierModel)
    this
  }

  def getBaseClassifier: OneVsRestTask = {
    if (classifier == "logisticRegression") {
      classifierModel = new LogisticRegressionTask(labelColumn, featureColumn, predictionColumn).defineEstimator.getEstimator.asInstanceOf[Classifier[_,_,_]]
    } else if (classifier == "decisionTree") {
      classifierModel = new DecisionTreeTask(labelColumn, featureColumn, predictionColumn).defineEstimator.getEstimator.asInstanceOf[Classifier[_,_,_]]
    } else if (classifier == "linearSvc") {
      classifierModel = new LinearSvcTask(labelColumn, featureColumn, predictionColumn).defineEstimator.getEstimator.asInstanceOf[Classifier[_,_,_]]
    } else if (classifier == "randomForest") {
      classifierModel = new RandomForestTask(labelColumn, featureColumn, predictionColumn).defineEstimator.getEstimator.asInstanceOf[Classifier[_,_,_]]
    } else if (classifier == "gbtClassifier") {
      classifierModel = new GbtClassifierTask(labelColumn, featureColumn, predictionColumn).defineEstimator.getEstimator.asInstanceOf[Classifier[_,_,_]]
    } else if (classifier == "naiveBayes") {
      classifierModel = new NaiveBayesTask(labelColumn, featureColumn, predictionColumn).defineEstimator.getEstimator.asInstanceOf[Classifier[_,_,_]]
    }
    this
  }

  override def loadModel(path: String): OneVsRestTask = {
    model = OneVsRestModel.load(path)
    this
  }

  override def saveModel(path: String): OneVsRestTask = {
    model.asInstanceOf[OneVsRestModel].write.overwrite().save(path)
    this
  }

}
