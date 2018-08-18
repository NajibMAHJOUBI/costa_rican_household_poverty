package fr.poverty.spark.sampleMethod.upSampleMinorityClass

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.Row
import scala.util.Random

/**
  * Down-sampling: process of randomly removing observations from the majoroty class in order to prevent its signal from dominating
  * the learning algorithm
  *
  * resample without replacement
  *
  */

class DownSampleMinorityClassTask(val labelColumn: String) {

  def run(spark: SparkSession, data: DataFrame): DataFrame = {
    resampleDataSet(spark, data, countByClass(spark, data), getSmallSize(countByClass(spark, data)))
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

  def getSmallSize(mapCountClass: Map[Int, Long]): Long = {
    mapCountClass.maxBy(_._2)._2
  }

  def resampleClass(spark: SparkSession, data: DataFrame, classIdentifier: Int, objectiveSize: Long): DataFrame = {
    val schema = data.schema
    val classBroadcast = spark.sparkContext.broadcast(classIdentifier)
    val filterClass = udf((label: Int) => label == classBroadcast.value)
    val rddClass: Array[Row] = data.filter(filterClass(col(labelColumn))).rdd.collect()
    val currentSize: Int  = rddClass.length
    var sampling: List[Row] = rddClass.toList
    if (currentSize > objectiveSize){
      val sliceIndex = Random.shuffle((0 until currentSize).toList).slice(0, objectiveSize.toInt)
      sliceIndex.foreach(index => {
        sampling = sampling ++ List(rddClass(index))
      })
    }
    val rdd = spark.sparkContext.parallelize(sampling)
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
