package fr.poverty.spark.classification.gridParameters

object GridParametersNaiveBayes {

  def getModelType(bernoulliOption: Boolean): Array[String] = {
    var params = Array("multinomial")
    if (bernoulliOption) {
      params = params ++ Array("bernoulli")
    }
    params
  }
}