package fr.poverty.spark.classification.task

import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}

class RandomForestTask(override val labelColumn: String, override val featureColumn: String, override val predictionColumn: String)
  extends ClassificationModelTask(labelColumn, featureColumn, predictionColumn)
    with ClassificationModelFactory {

  override def defineEstimator: RandomForestTask= {
    estimator = new RandomForestClassifier().setFeaturesCol(featureColumn).setLabelCol(labelColumn).setPredictionCol(predictionColumn)
    this
  }

  override def loadModel(path: String): RandomForestTask = {
    model = RandomForestClassificationModel.load(path)
    this
  }

  override def saveModel(path: String): RandomForestTask = {
    model.asInstanceOf[RandomForestClassificationModel].write.overwrite().save(path)
    this
  }

}
