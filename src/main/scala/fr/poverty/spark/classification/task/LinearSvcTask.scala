package fr.poverty.spark.classification.task

import org.apache.spark.ml.classification.{LinearSVC, LinearSVCModel}

/**
  * Created by mahjoubi on 12/06/18.
  *
  * LinearSVC classifier
  *
  */
class LinearSvcTask(override val labelColumn: String, override val featureColumn: String, override val predictionColumn: String)
  extends ClassificationModelTask(labelColumn, featureColumn, predictionColumn)
    with ClassificationModelFactory {

  override def defineEstimator: LinearSvcTask= {
    estimator = new LinearSVC().setFeaturesCol(featureColumn).setLabelCol(labelColumn).setPredictionCol(predictionColumn)
    this
  }

  override def loadModel(path: String): LinearSvcTask = {
    model = LinearSVCModel.load(path)
    this
  }

  override def saveModel(path: String): LinearSvcTask = {
    model.asInstanceOf[LinearSVCModel].write.overwrite().save(path)
    this
  }

}
