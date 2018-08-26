package fr.poverty.spark.classification.task


trait ClassificationModelFactory {

  def defineEstimator: ClassificationModelFactory

  def saveModel(path: String): ClassificationModelFactory

  def loadModel(path: String): ClassificationModelFactory

}
