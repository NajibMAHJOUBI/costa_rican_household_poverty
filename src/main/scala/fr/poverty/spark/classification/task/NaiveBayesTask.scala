package fr.poverty.spark.classification.task

import org.apache.spark.ml.classification.{NaiveBayes, NaiveBayesModel}

class NaiveBayesTask(override val labelColumn: String, override val featureColumn: String, override val predictionColumn: String)
  extends ClassificationModelTask(labelColumn, featureColumn, predictionColumn)
    with ClassificationModelFactory {

  override def defineEstimator: NaiveBayesTask= {
    estimator = new NaiveBayes().setFeaturesCol(featureColumn).setLabelCol(labelColumn).setPredictionCol(predictionColumn)
    this
  }

  override def loadModel(path: String): NaiveBayesTask = {
    model = NaiveBayesModel.load(path)
    this
  }

  override def saveModel(path: String): NaiveBayesTask = {
    model.asInstanceOf[NaiveBayesModel].write.overwrite().save(path)
    this
  }

}
