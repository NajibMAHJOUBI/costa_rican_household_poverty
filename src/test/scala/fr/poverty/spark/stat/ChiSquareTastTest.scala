package fr.poverty.spark.stat

import fr.poverty.spark.utils.LoadDataSetTask
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}

import scala.io.Source

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
    val categoricalFeatures = Source
      .fromFile("/home/mahjoubi/Documents/github/costa_rican_household_poverty/src/main/resources/categoricalFeatures")
      .getLines
      .toList
      .head
      .split(",").toList
    val data = new LoadDataSetTask(sourcePath = "data", format = "csv").run(spark, "train")
//    val featuresList = List("paredblolad", "paredzocalo", "paredpreb", "pareddes", "paredmad", "paredzinc", "paredfibras", "paredother")
    val chiSelector = new ChiSquareTask("Target", categoricalFeatures, 0.05)

    val vectorSize = udf((x: Vector) => x.size)

    val selected = chiSelector.run(spark, data)
      .withColumn("featuresSize", vectorSize(col("features")))
      .withColumn("selectedSize", vectorSize(col("selectedFeatures")))

    selected.show(false)
  }


  @After def afterAll() {
    spark.stop()
  }
}


