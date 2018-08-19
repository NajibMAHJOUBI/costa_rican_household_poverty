package fr.poverty.spark.sampleMinorityClass

import org.apache.spark.sql.{DataFrame, SparkSession}

class SampleMinorityClassTask(val labelColumn: String) {

  def dataSetClass(spark: SparkSession, data: DataFrame): List[Int] = {
    val labelBroadcast = spark.sparkContext.broadcast(labelColumn)
    data.select(labelColumn).distinct()
      .rdd.map(line => line.getInt(line.fieldIndex(labelBroadcast.value)))
      .collect().toList
  }

  def countByClass(spark: SparkSession, data: DataFrame): Map[Int, Long] = {
    val labelBroadcast = spark.sparkContext.broadcast(labelColumn)
    data.groupBy(labelColumn).count()
      .rdd.map(row => (row.getInt(row.fieldIndex(labelBroadcast.value)), row.getLong(row.fieldIndex("count"))))
      .collectAsMap().toMap
  }
}
