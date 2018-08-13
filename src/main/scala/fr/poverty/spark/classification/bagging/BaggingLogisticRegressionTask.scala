package fr.poverty.spark.classification.bagging

import org.apache.spark.sql.DataFrame

class BaggingLogisticRegressionTask(override val idColumn: String, override val labelColumn: String,
                                    override val featureColumn: String, override val predictionColumn: String,
                                    override val pathSave: String,
                                    override val numberOfSampling: Int, override val samplingFraction: Double,
                                    override val validationMethod: String, override val ratio: Double) extends
  BaggingTask(idColumn, labelColumn, featureColumn, predictionColumn, pathSave, numberOfSampling,
    samplingFraction, validationMethod, ratio) {

  def run(data: DataFrame): Unit = {

  }

}
