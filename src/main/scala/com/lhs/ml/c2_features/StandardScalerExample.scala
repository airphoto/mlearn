package com.lhs.ml.c2_features

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

/**
  * StandardScaler转换Vector行的数据集，使每个要素标准化以具有单位标准偏差和 或 零均值。它需要参数：
  *   * withStd：默认为True。将数据缩放到单位标准偏差。
  *   * withMean：默认为false。在缩放之前将数据中心为平均值。它将构建一个密集的输出，所以在应用于稀疏输入时要小心。
  * StandardScaler是一个Estimator，可以适合数据集生成StandardScalerModel; 这相当于计算汇总统计数据。 \
  * 然后，模型可以将数据集中的向量列转换为具有单位标准偏差和/或零平均特征。
  *
  * 请注意，如果特征的标准偏差为零，它将在该特征的向量中返回默认的0.0值。
  * 以下示例演示如何以libsvm格式加载数据集，然后将每个要素归一化以具有单位标准偏差。
  */
object StandardScalerExample {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val spark = SparkSession.builder().master("local").getOrCreate()
    import spark.implicits._
    val dataFrame = spark.createDataFrame(Seq(
      (0, Vectors.dense(1.0, 0.5, -1.0)),
      (1, Vectors.dense(2.0, 1.0, 1.0)),
      (2, Vectors.dense(4.0, 10.0, 2.0))
    )).toDF("id", "features")

    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeature")

    val scalerModel = scaler.fit(dataFrame)
    val scalerData = scalerModel.transform(dataFrame)


    scalerData.show(false)
  }
}
