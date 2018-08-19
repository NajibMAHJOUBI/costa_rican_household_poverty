package fr.poverty.spark.classification.ensembleMethod.bagging

import fr.poverty.spark.classification.validation.crossValidation.CrossValidationGbtClassifierTask
import fr.poverty.spark.classification.validation.trainValidation.TrainValidationGbtClassifierTask
import org.apache.spark.ml.classification.GBTClassificationModel
import org.apache.spark.sql.DataFrame

class BaggingGbtClassifierTask(override val idColumn: String, override val labelColumn: String,
                               override val featureColumn: String, override val predictionColumn: String,
                               override val pathSave: String,
                               override val numberOfSampling: Int, override val samplingFraction: Double,
                               override val validationMethod: String, override val ratio: Double) extends
  BaggingTask(idColumn, labelColumn, featureColumn, predictionColumn, pathSave, numberOfSampling,
    samplingFraction, validationMethod, ratio) {

  var modelFittedList: List[GBTClassificationModel] = List()

  def run(data: DataFrame): Unit = {
    defineSampleSubset(data)
    loopDataSampling()
  }

  def loopDataSampling(): Unit = {
    sampleSubsetsList.foreach(sample => {
      if(validationMethod == "trainValidation"){
        val trainValidation = new TrainValidationGbtClassifierTask(labelColumn, featureColumn, predictionColumn, "", ratio)
        trainValidation.run(sample)
        modelFittedList = modelFittedList ++ List(trainValidation.getBestModel)
      } else if(validationMethod == "crossValidation"){
        val crossValidation = new CrossValidationGbtClassifierTask(labelColumn, featureColumn, predictionColumn, "", ratio.toInt)
        crossValidation.run(sample)
        modelFittedList = modelFittedList ++ List(crossValidation.getBestModel)
      }
    })
  }

  def getModels: List[GBTClassificationModel] = modelFittedList
}
