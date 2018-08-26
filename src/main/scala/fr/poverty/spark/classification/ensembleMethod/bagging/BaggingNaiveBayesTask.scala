package fr.poverty.spark.classification.ensembleMethod.bagging

import fr.poverty.spark.classification.validation.crossValidation.CrossValidationNaiveBayesTask
import fr.poverty.spark.classification.validation.trainValidation.TrainValidationNaiveBayesTask
import org.apache.spark.ml.classification.NaiveBayesModel
import org.apache.spark.sql.DataFrame

class BaggingNaiveBayesTask(override val idColumn: String,
                            override val labelColumn: String,
                            override val featureColumn: String,
                            override val predictionColumn: String,
                            override val pathSave: String,
                            override val numberOfSampling: Int,
                            override val samplingFraction: Double,
                            override val validationMethod: String,
                            override val ratio: Double,
                            override val metricName: String,
                            val bernoulliOption: Boolean)
  extends BaggingTask(idColumn, labelColumn, featureColumn, predictionColumn, pathSave, numberOfSampling, samplingFraction, validationMethod, ratio, metricName)
with BaggingModelFactory {

  var modelList: List[NaiveBayesModel] = List()

  def run(data: DataFrame): Unit = {
    defineSampleSubset(data)
    loopDataSampling()
  }

  def loopDataSampling(): Unit = {
    sampleSubsetsList.foreach(sample => {
      if(validationMethod == "trainValidation"){
        val trainValidation = new TrainValidationNaiveBayesTask(labelColumn, featureColumn, predictionColumn, metricName, pathSave, ratio, bernoulliOption)
        trainValidation.run(sample)
        modelList = modelList ++ List(trainValidation.getBestModel.asInstanceOf[NaiveBayesModel])
      } else if(validationMethod == "crossValidation"){
        val crossValidation = new CrossValidationNaiveBayesTask(labelColumn, featureColumn, predictionColumn, metricName, pathSave, ratio.toInt, bernoulliOption)
        crossValidation.run(sample)
        modelList = modelList ++ List(crossValidation.getBestModel.asInstanceOf[NaiveBayesModel])
      }
    })
  }

  def getModels: List[NaiveBayesModel] = modelList
}
