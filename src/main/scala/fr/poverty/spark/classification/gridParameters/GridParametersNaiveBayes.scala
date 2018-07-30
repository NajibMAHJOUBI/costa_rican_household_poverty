package fr.poverty.spark.classification.gridParameters

import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.ParamGridBuilder

object GridParametersNaiveBayes {

  def getModelType(bernoulliOption: Boolean): Array[String] = {
    var params = Array("multinomial")
    if (bernoulliOption) {
      params = params ++ Array("bernoulli")
    }
    params
  }

  def getParamsGrid(estimator: NaiveBayes, bernoulliOption: Boolean): Array[ParamMap] = {
    new ParamGridBuilder()
      .addGrid(estimator.modelType, getModelType(bernoulliOption))
      .build()
  }

}