package fr.poverty.spark.stat

import fr.poverty.spark.sampleMinorityClass.SampleMinorityClassTask
import fr.poverty.spark.utils.LoadDataSetTask
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.types.{DoubleType, IntegerType, StructField, StructType}
import org.junit.{After, Before, Test}

class ChiSquareTastTest {

  private val labelColumn: String = "label"
  private var spark: SparkSession = _
  private var data: DataFrame = _

  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("chi square suite test")
      .getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)
  }

  @Test def testChiSquare(): Unit = {
    val data = new LoadDataSetTask(sourcePath = "data", format = "csv").run(spark, "train")
    val featuresList = List("paredblolad", "paredzocalo", "paredpreb", "pareddes", "paredmad", "paredzinc", "paredfibras", "paredother")
    val chiSelector = new ChiSquareTask("Target", featuresList, 0.05)
    chiSelector.run(spark, data).show(false)
  }


  @After def afterAll() {
    spark.stop()
  }
}


