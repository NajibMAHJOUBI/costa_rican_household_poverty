package fr.poverty.spark.classification.ensembleMethod.bagging

import fr.poverty.spark.classification.validation.crossValidation.CrossValidationOneVsRestTask
import fr.poverty.spark.classification.validation.trainValidation.TrainValidationOneVsRestTask
import org.apache.spark.ml.classification.OneVsRestModel
import org.apache.spark.sql.DataFrame

class BaggingOneVsRestTask(override val idColumn: String,
                           override val labelColumn: String,
                           override val featureColumn: String,
                           override val predictionColumn: String,
                           override val pathSave: String,
                           override val numberOfSampling: Int,
                           override val samplingFraction: Double,
                           override val validationMethod: String,
                           override val ratio: Double,
                           override val metricName: String,
                           val classifier: String,
                           val bernoulliOption: Boolean = false)
  extends BaggingTask(idColumn, labelColumn, featureColumn, predictionColumn, pathSave, numberOfSampling, samplingFraction, validationMethod, ratio, metricName)
  with BaggingModelFactory {

  var modelFittedList: List[OneVsRestModel] = List()

  override def run(data: DataFrame): Unit = {
    defineSampleSubset(data)
    loopDataSampling()
  }

  override def loopDataSampling(): Unit = {
    sampleSubsetsList.foreach(sample => {
      if(validationMethod == "trainValidation"){
        val trainValidation = new TrainValidationOneVsRestTask(labelColumn, featureColumn, predictionColumn, metricName, pathSave, ratio, classifier, bernoulliOption)
        trainValidation.run(sample)
        modelFittedList = modelFittedList ++ List(trainValidation.getBestModel)
      } else if(validationMethod == "crossValidation"){
        val crossValidation = new CrossValidationOneVsRestTask(labelColumn, featureColumn, predictionColumn, metricName, pathSave, ratio.toInt, classifier, bernoulliOption)
        crossValidation.run(sample)
        modelFittedList = modelFittedList ++ List(crossValidation.getBestModel)
      }
    })
  }

  override def getModels: List[OneVsRestModel] = modelFittedList
}
