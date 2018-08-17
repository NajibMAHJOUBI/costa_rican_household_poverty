package fr.poverty.spark.sampleMethod.upSampleMinorityClass

import org.apache.spark.sql.DataFrame

/**
  * Up-sampling: process of randomly duplicating observations from the minority class in order to reinforce its signal
  *
  * resample with replacement
  *
  */

class UpSampleMinorityClassTask(val labelColumn: String) {



  def dataSetClass(data: DataFrame): List[String] = {
    data.select(labelColumn).distinct()
      .rdd.map(line => line.getString(line.fieldIndex(labelColumn)))
      .collect().toList
  }

  def countByClass(data: DataFrame): Map[String, Long] = {
    data.groupBy(labelColumn).count()
      .rdd.map(row => (row.getString(row.fieldIndex(labelColumn)), row.getLong(row.fieldIndex("count"))))
      .collectAsMap().toMap
  }
}
