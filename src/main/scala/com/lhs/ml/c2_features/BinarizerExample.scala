package com.lhs.ml.c2_features

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.Binarizer
import org.apache.spark.sql.SparkSession

/**
  * Binarizer（ML提供的二元化方法）二元化涉及的参数有 inputCol（输入）、outputCol（输出）以及threshold（阀值）。
  * （输入的）特征值大于阀值将二值化为1.0，特征值小于等于阀值将二值化为0.0。inputCol 支持向量（Vector）和双精度（Double）类型。
  */
object BinarizerExample {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val spark = SparkSession.builder().master("local").getOrCreate()
    import spark.implicits._

    val data = Array((0, 0.1), (1, 0.8), (2, 0.2))
    val dataFrame = spark.createDataFrame(data).toDF("id","features")

    val binarizer = new Binarizer().setInputCol("features").setOutputCol("binary_feature").setThreshold(0.5)

    val binarizedDataFrame = binarizer.transform(dataFrame)

    binarizedDataFrame.show(false)
  }
}
