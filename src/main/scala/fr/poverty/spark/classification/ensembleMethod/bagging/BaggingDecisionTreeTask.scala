package fr.poverty.spark.classification.ensembleMethod.bagging

import fr.poverty.spark.classification.validation.crossValidation.CrossValidationDecisionTreeTask
import fr.poverty.spark.classification.validation.trainValidation.TrainValidationDecisionTreeTask
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.sql.DataFrame

class BaggingDecisionTreeTask(override val idColumn: String,
                              override val labelColumn: String,
                              override val featureColumn: String,
                              override val predictionColumn: String,
                              override val pathSave: String,
                              override val numberOfSampling: Int,
                              override val samplingFraction: Double,
                              override val validationMethod: String,
                              override val ratio: Double,
                              override val metricName: String)
  extends BaggingTask(idColumn, labelColumn, featureColumn, predictionColumn, pathSave, numberOfSampling, samplingFraction, validationMethod, ratio, metricName)
  with BaggingModelFactory {

  var modelList: List[DecisionTreeClassificationModel] = List()

  override def run(data: DataFrame): Unit = {
    defineSampleSubset(data)
    loopDataSampling()
  }

  override def loopDataSampling(): Unit = {
    sampleSubsetsList.foreach(sample => {
      if(validationMethod == "trainValidation"){
        val trainValidation = new TrainValidationDecisionTreeTask(labelColumn, featureColumn, predictionColumn, metricName, pathSave, ratio)
        trainValidation.run(sample)
        modelList = modelList ++ List(trainValidation.getBestModel.asInstanceOf[DecisionTreeClassificationModel])
      } else if(validationMethod == "crossValidation"){
        val crossValidation = new CrossValidationDecisionTreeTask(labelColumn, featureColumn, predictionColumn, metricName, pathSave, ratio.toInt)
        crossValidation.run(sample)
        modelList = modelList ++ List(crossValidation.getBestModel.asInstanceOf[DecisionTreeClassificationModel])
      }
    })
  }

  def getModels: List[DecisionTreeClassificationModel] = modelList
}
