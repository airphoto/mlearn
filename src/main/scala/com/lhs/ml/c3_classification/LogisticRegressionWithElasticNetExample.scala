package com.lhs.ml.c3_classification

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object LogisticRegressionWithElasticNetExample {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val spark = SparkSession.builder().master("local").getOrCreate()
    import spark.implicits._

    val training = spark.read.format("libsvm").load("E:\\sources\\spark-2.3.0\\data\\mllib\\sample_libsvm_data.txt")
    training.show(false)
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    val lrModel = lr.fit(training)

    val mlr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
      .setFamily("multinomial")

    val mlrModel = mlr.fit(training)

    println(s"每个特征的系数: ${lrModel.coefficients} \n截距: ${lrModel.intercept}")
    println(s"Multinomial 每个特征的系数: ${mlrModel.coefficientMatrix}")
    println(s"Multinomial 截距: ${mlrModel.interceptVector}")


    val trainingSummary = lrModel.summary

    //每个迭代的目标
    val objectiveHistory = trainingSummary.objectiveHistory
    println("每次迭代的目标")
    objectiveHistory.foreach(println)

    //获取用于判断测试数据性能的指标。
    //逻辑回归的spark.ml实现也支持在训练集中提取模型的摘要。
    // 请注意，在BinaryLogisticRegressionSummary中存储为DataFrame的预测和度量标注为@transient，因此仅在驱动程序上可用。
    //LogisticRegressionTrainingSummary为LogisticRegressionModel提供了一个摘要。
    // 目前，只支持二进制分类，必须将摘要显式转换为BinaryLogisticRegressionTrainingSummary。 当支持多类分类时，这可能会发生变化。
    //将summary转化为 BinaryLogisticRegressionSummary  ， 因为这是个二分类的问题
    val binarySummary = trainingSummary.asInstanceOf[BinaryLogisticRegressionSummary]

    val roc = binarySummary.roc
    roc.show(false)
    println(s"roc 下的面积 ${binarySummary.areaUnderROC}")

    val fMeasure = binarySummary.fMeasureByThreshold
    val maxFMeasure = fMeasure.select(max($"f-Measure")).head().getDouble(0)
    val bestThreshold = fMeasure.where($"F-Measure" === maxFMeasure).select("threshold").head().getDouble(0)

    fMeasure.show(false)
  }
}
