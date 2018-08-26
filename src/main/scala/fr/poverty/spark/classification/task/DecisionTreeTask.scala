package fr.poverty.spark.classification.task

import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}

class DecisionTreeTask(override val labelColumn: String, override val featureColumn: String, override val predictionColumn: String)
  extends ClassificationModelTask(labelColumn, featureColumn, predictionColumn)
    with ClassificationModelFactory {

  override def defineEstimator: DecisionTreeTask= {
    estimator = new DecisionTreeClassifier()
      .setFeaturesCol(featureColumn)
      .setLabelCol(labelColumn)
      .setPredictionCol(predictionColumn)
    this
  }

  override def loadModel(path: String): DecisionTreeTask = {
    model = DecisionTreeClassificationModel.load(path)
    this
  }

  override def saveModel(path: String): DecisionTreeTask = {
    model.asInstanceOf[DecisionTreeClassificationModel].write.overwrite().save(path)
    this
  }

}
