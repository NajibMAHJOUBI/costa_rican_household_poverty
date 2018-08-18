package fr.poverty.spark.sampleMethod.upSampleMinorityClass

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.Row
import java.util.Random

/**
  * Up-sampling: process of randomly duplicating observations from the minority class in order to reinforce its signal
  *
  * resample with replacement
  *
  */

class UpSampleMinorityClassTask(val labelColumn: String) {

  def run(spark: SparkSession, data: DataFrame): DataFrame = {
    resampleDataSet(spark, data, countByClass(spark, data), getLargeSize(countByClass(spark, data)))
  }

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

  def getLargeSize(mapCountClass: Map[Int, Long]): Long = {
    mapCountClass.maxBy(_._2)._2
  }

  def resampleClass(spark: SparkSession, data: DataFrame, classIdentifier: Int, objectiveSize: Long): DataFrame = {
    val schema = data.schema
    val classBroadcast = spark.sparkContext.broadcast(classIdentifier)
    val filterClass = udf((label: Int) => label == classBroadcast.value)
    val rddClass: Array[Row] = data.filter(filterClass(col(labelColumn))).rdd.collect()
    val currentSize: Int  = rddClass.length
    var sampling: List[Row] = rddClass.toList
    if (objectiveSize - currentSize > 0){
      val random = new Random()
      (0 until objectiveSize.toInt - currentSize).foreach(_ => {
        sampling = sampling ++ List(rddClass(random.nextInt(currentSize)))
      })
    }
    val rdd = spark.sparkContext.parallelize(sampling.toSeq)
    spark.createDataFrame(rdd, schema)
  }

  def resampleDataSet(spark: SparkSession, data: DataFrame, mapClassCount: Map[Int, Long], objectiveSize: Long): DataFrame = {
     var sampleList: List[DataFrame] = List()
     mapClassCount.map(_._1).foreach(classIdentifier => {
       sampleList = sampleList ++ List(resampleClass(spark, data, classIdentifier, objectiveSize))
     })
     var newDataFrame = sampleList(0)
     (1 until sampleList.length).foreach(index => newDataFrame = newDataFrame.union(sampleList(index)))
     newDataFrame
  }

}
