package fr.poverty.spark.classification.validation

import org.apache.spark.sql.DataFrame

object ValidationObject {

  def trainValidationSplit(data: DataFrame, ratio: Double): Array[DataFrame] = {
    data.randomSplit(Array(ratio, (1-ratio)))
  }

}
