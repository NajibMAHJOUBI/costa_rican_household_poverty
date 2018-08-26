package fr.poverty.spark.stat

import fr.poverty.spark.utils.LoadDataSetTask
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}

import scala.io.Source

class ChiSquareTastTest {

  private var spark: SparkSession = _

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
    val train = new LoadDataSetTask(sourcePath = "data", format = "csv").run(spark, "train")
    val test = new LoadDataSetTask(sourcePath = "data", format = "csv").run(spark, "test")
    val featuresColumn = List("paredblolad", "paredzocalo", "paredpreb", "pareddes", "paredmad", "paredzinc", "paredfibras", "paredother")
    val chiSelector = new ChiSquareTask("Target", "Target", categoricalFeatures, "feature",0.05)
//    val idColumn: String, val labelColumn: String, val featuresColumn: List[String], val featureColumn: String, alpha: Double
    chiSelector.run(spark, train, test)

    val chiTrain = chiSelector.transform(train, "train")
    val chiTest = chiSelector.transform(train, "test")

    assert(chiTrain.isInstanceOf[DataFrame])
    assert(chiTest.isInstanceOf[DataFrame])
  }


  @After def afterAll() {
    spark.stop()
  }
}


