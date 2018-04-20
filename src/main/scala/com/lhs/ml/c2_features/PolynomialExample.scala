package com.lhs.ml.c2_features

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.PolynomialExpansion
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

/**
  * （多项式展开）是将特征扩展为多项式空间的过程，多项式空间由原始维度的n度组合组成。
  * PolynomialExpansion类提供此功能。
  * 下面的例子显示了如何将您的功能扩展到3度多项式空间。
  */
object PolynomialExample {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val spark = SparkSession.builder().master("local").getOrCreate()
    import spark.implicits._

    val data = Array(
      Vectors.dense(2.0, 1.0),
      Vectors.dense(0.0, 0.0),
      Vectors.dense(3.0, -1.0)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply(_))).toDF("features")

    val polyExpansion = new PolynomialExpansion()
      .setInputCol("features")
      .setOutputCol("polyFeatures")
      .setDegree(3)

    val polyDF = polyExpansion.transform(df)

    polyDF.show(false)

  }
}
