package fr.poverty.spark.classification.task

import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}

/**
  * Created by mahjoubi on 12/06/18.
  */
class LogisticRegressionTask(override val labelColumn: String, override val featureColumn: String, override val predictionColumn: String)
  extends ClassificationModelTask(labelColumn, featureColumn, predictionColumn)
    with ClassificationModelFactory {

  override def defineEstimator: LogisticRegressionTask= {
    estimator = new LogisticRegression().setFeaturesCol(featureColumn).setLabelCol(labelColumn).setPredictionCol(predictionColumn)
    this
  }

  override def loadModel(path: String): LogisticRegressionTask = {
    model = LogisticRegressionModel.load(path)
    this
  }

  override def saveModel(path: String): LogisticRegressionTask = {
    model.asInstanceOf[LogisticRegressionModel].write.overwrite().save(path)
    this
  }



}
