package fr.poverty.spark.classification.ensembleMethod.bagging

import fr.poverty.spark.classification.validation.crossValidation.CrossValidationDecisionTreeTask
import fr.poverty.spark.classification.validation.trainValidation.TrainValidationDecisionTreeTask
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.sql.DataFrame

class BaggingDecisionTreeTask(override val idColumn: String, override val labelColumn: String,
                              override val featureColumn: String, override val predictionColumn: String,
                              override val pathSave: String,
                              override val numberOfSampling: Int, override val samplingFraction: Double,
                              override val validationMethod: String, override val ratio: Double) extends
  BaggingTask(idColumn, labelColumn, featureColumn, predictionColumn, pathSave, numberOfSampling,
    samplingFraction, validationMethod, ratio) {

  var modelFittedList: List[DecisionTreeClassificationModel] = List()

  def run(data: DataFrame): Unit = {
    defineSampleSubset(data)
    loopDataSampling()
  }

  def loopDataSampling(): Unit = {
    sampleSubsetsList.foreach(sample => {
      if(validationMethod == "trainValidation"){
        val trainValidation = new TrainValidationDecisionTreeTask(labelColumn, featureColumn, predictionColumn, "", ratio)
        trainValidation.run(sample)
        modelFittedList = modelFittedList ++ List(trainValidation.getBestModel)
      } else if(validationMethod == "crossValidation"){
        val crossValidation = new CrossValidationDecisionTreeTask(labelColumn, featureColumn, predictionColumn, "", ratio.toInt)
        crossValidation.run(sample)
        modelFittedList = modelFittedList ++ List(crossValidation.getBestModel)
      }
    })
  }

  def getModels: List[DecisionTreeClassificationModel] = modelFittedList
}
