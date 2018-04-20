package com.lhs.ml.c1_pipeline

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{IndexToString, StringIndexer}
import org.apache.spark.sql.SparkSession

object Index2String {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val spark = SparkSession.builder().master("local").getOrCreate()
    import spark.implicits._

    val df = spark.createDataFrame(Seq(
      (0, "a"),
      (1, "b"),
      (2, "c"),
      (3, "a"),
      (4, "a"),
      (5, "c")
    )).toDF("id", "category")

    val indexer = new StringIndexer()
      .setInputCol("category")
      .setOutputCol("categoryIndex")

    val index2String = new IndexToString()
      .setInputCol(indexer.getOutputCol)
      .setOutputCol("categories")

    val pipeline = new Pipeline().setStages(Array(indexer,index2String))

    val model = pipeline.fit(df)

    val data = model.transform(df)

    data.show(false)
    spark.stop()
  }
}
