package fr.poverty.spark.utils

object UtilsObject {

  def dealNullValue(x: Option[Double]): Option[Double] = {
    val num = x.getOrElse(return None)
    Some(num)
  }

}
