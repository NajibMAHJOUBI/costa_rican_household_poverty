package fr.poverty.spark.classification.ensembleMethod.bagging

import fr.poverty.spark.classification.validation.crossValidation.CrossValidationRandomForestTask
import fr.poverty.spark.classification.validation.trainValidation.TrainValidationRandomForestTask
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.sql.DataFrame

class BaggingRandomForestTask(override val idColumn: String,
                              override val labelColumn: String,
                              override val featureColumn: String,
                              override val predictionColumn: String,
                              override val pathSave: String,
                              override val numberOfSampling: Int,
                              override val samplingFraction: Double,
                              override val validationMethod: String,
                              override val ratio: Double,
                              override val metricName: String)
  extends BaggingTask(idColumn, labelColumn, featureColumn, predictionColumn, pathSave, numberOfSampling,
    samplingFraction, validationMethod, ratio, metricName)
with BaggingModelFactory {

  var modelFittedList: List[RandomForestClassificationModel] = List()

  override def run(data: DataFrame): Unit = {
    defineSampleSubset(data)
    loopDataSampling()
  }

  override def loopDataSampling(): Unit = {
    sampleSubsetsList.foreach(sample => {
      if(validationMethod == "trainValidation"){
        val trainValidation = new TrainValidationRandomForestTask(labelColumn, featureColumn, predictionColumn, metricName, pathSave, ratio)
        trainValidation.run(sample)
        modelFittedList = modelFittedList ++ List(trainValidation.getBestModel)
      } else if(validationMethod == "crossValidation"){
        val crossValidation = new CrossValidationRandomForestTask(labelColumn, featureColumn, predictionColumn, metricName, pathSave, ratio.toInt)
        crossValidation.run(sample)
        modelFittedList = modelFittedList ++ List(crossValidation.getBestModel)
      }
    })
  }

  override def getModels: List[RandomForestClassificationModel] = modelFittedList
}
