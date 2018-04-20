package com.lhs.ml.c1_pipeline

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{OneHotEncoderEstimator, StringIndexer}
import org.apache.spark.sql.SparkSession

/**
  * 独热编码即 One-Hot 编码，又称一位有效编码，
  * 其方法是使用N位状态寄存器来对N个状态进行编码，每个状态都由他独立的寄存器位，并且在任意时候，其中只有一位有效。
  * 可以这样理解，对于每一个特征，如果它有m个可能值，那么经过独热编码后，就变成了m个二元特征。
  * 并且，这些特征互斥，每次只有一个激活。因此，数据会变成稀疏的。
  *
  * 由第一部分的分析，很容易看出one hot编码的优点：
  * 1.能够处理非连续型数值特征。
  * 2.在一定程度上也扩充了特征。比如性别本身是一个特征，经过one hot编码以后，就变成了男或女两个特征。
  */
object OneHotEncoderExample {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val spark = SparkSession.builder().master("local").getOrCreate()
    import spark.implicits._

    val df = spark.createDataFrame(Seq(
      (0, "a","one"),
      (1, "b","two"),
      (2, "c","three"),
      (3, "a","one"),
      (4, "a","one"),
      (5, "c","two"),
      (6, "d","two")
    )).toDF("id", "category","ins")

    val categoryIndexer = new StringIndexer()
      .setInputCol("category")
      .setOutputCol("categoryIndex")

    val insCategory = new StringIndexer()
      .setInputCol("ins")
      .setOutputCol("insIndex")

    val encoder = new OneHotEncoderEstimator()
      .setInputCols(Array("categoryIndex","insIndex"))
      .setOutputCols(Array("categoryEncoder","insEncoder"))

    val pipeline = new Pipeline().setStages(Array(categoryIndexer,insCategory,encoder))

    val model = pipeline.fit(df)

    val data = model.transform(df)

    data.show(false)
    data.select($"categoryEncoder",$"insEncoder").show(false)
  }
}
