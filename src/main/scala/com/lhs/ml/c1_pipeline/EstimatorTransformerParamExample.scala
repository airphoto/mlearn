package com.lhs.ml.c1_pipeline

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.{Row, SparkSession}

object EstimatorTransformerParamExample {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val spark = SparkSession.builder().master("local").getOrCreate()
    import spark.implicits._

    val training = spark.createDataFrame(Seq(
      (1.0, Vectors.dense(0.0, 1.1, 0.1)),
      (0.0, Vectors.dense(2.0, 1.0, -1.0)),
      (0.0, Vectors.dense(2.0, 1.3, 1.0)),
      (1.0, Vectors.dense(0.0, 1.2, -0.5))
    )).toDF("label", "features")

    val lr = new LogisticRegression()
    println("逻辑回归的各个参数：\n"+lr.explainParams()+"\n")
    lr.setMaxIter(10).setRegParam(0.01)

    val model = lr.fit(training)
    println("model 1 使用的参数有："+model.parent.extractParamMap())

    val paramMap = ParamMap(lr.maxIter->20)
      .put(lr.maxIter,30)   //设置一个参数
      .put(lr.regParam->0.1,lr.threshold->0.55) //设置多个参数

    val param2 = ParamMap(lr.probabilityCol->"myProbability")

    //可以把参数合并
    val paramMapCombine = param2 ++ paramMap

    val model2 = lr.fit(training,paramMapCombine)

    println("Model 2 使用的参数有: " + model2.parent.extractParamMap)

    val test = spark.createDataFrame(Seq(
      (1.0, Vectors.dense(-1.0, 1.5, 1.3)),
      (0.0, Vectors.dense(3.0, 2.0, -0.1)),
      (1.0, Vectors.dense(0.0, 2.2, -1.5))
    )).toDF("label", "features")

    model2.transform(test)
      .select("features", "label", "myProbability", "prediction")
      .collect()
      .foreach { case Row(features: Vector, label: Double, prob: Vector, prediction: Double) =>
        println(s"($features, $label) -> prob=$prob, prediction=$prediction")
      }

    spark.close()
  }
}
