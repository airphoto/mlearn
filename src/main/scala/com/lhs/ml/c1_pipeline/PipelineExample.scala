package com.lhs.ml.c1_pipeline

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.sql.{Row, SparkSession}

object PipelineExample {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val spark = SparkSession.builder().master("local").getOrCreate()
    import spark.implicits._

    val traning = spark.createDataFrame(Seq(
      (0L, "a b c d e spark", 1.0),
      (1L, "b d", 0.0),
      (2L, "spark f g h", 1.0),
      (3L, "hadoop mapreduce", 0.0)
    )).toDF("id","text","label")

    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")

    val hashingTF = new HashingTF().setNumFeatures(1000).setInputCol(tokenizer.getOutputCol).setOutputCol("features")

    val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.001)

    //pipeline 过程
    val pipeline = new Pipeline().setStages(Array(tokenizer,hashingTF,lr))

    //模型
    val model = pipeline.fit(traning)
    //存储模型
    model.write.overwrite().save("spark-logistic-regression-model")

    pipeline.write.overwrite().save("unfit-lr-model")
    //获取模型
    val sameModel = PipelineModel.load("spark-logistic-regression-model")

    val test = spark.createDataFrame(Seq(
      (4L, "spark i j k"),
      (5L, "l m n"),
      (6L, "spark hadoop spark"),
      (7L, "apache hadoop")
    )).toDF("id", "text")

    sameModel.transform(test).select("id", "text", "probability", "prediction")
      .collect()
      .foreach { case Row(id: Long, text: String, prob: Vector, prediction: Double) =>
        println(s"($id, $text) --> prob=$prob, prediction=$prediction")
      }

    spark.close()
  }
}
