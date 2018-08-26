package fr.poverty.spark.classification.task

import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}

/**
  * Created by mahjoubi on 12/06/18.
  */


class GbtClassifierTask(override val labelColumn: String, override val featureColumn: String, override val predictionColumn: String)
  extends ClassificationModelTask(labelColumn, featureColumn, predictionColumn)
    with ClassificationModelFactory {

  override def defineEstimator: GbtClassifierTask= {
    estimator = new GBTClassifier()
      .setFeaturesCol(featureColumn)
      .setLabelCol(labelColumn)
      .setPredictionCol(predictionColumn)
    this
  }

  override def loadModel(path: String): GbtClassifierTask = {
    model = GBTClassificationModel.load(path)
    this
  }

  override def saveModel(path: String): GbtClassifierTask = {
    model.asInstanceOf[GBTClassificationModel].write.overwrite().save(path)
    this
  }

}
