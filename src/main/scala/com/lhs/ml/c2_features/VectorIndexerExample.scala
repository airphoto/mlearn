package com.lhs.ml.c2_features

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.sql.SparkSession

/**
  * VectorIndexer可以帮助指定向量数据集中的分类特征。它可以自动确定哪些功能是分类的，并将原始值转换为类别索引。具体来说，它执行以下操作：
  * 取一个Vector类型的输入列和一个参数maxCategories。
  * 根据不同值的数量确定哪些功能应分类，其中最多maxCategories的功能被声明为分类。
  * 为每个分类功能计算基于0的类别索引。
  * 索引分类特征并将原始特征值转换为索引。
  * 索引分类功能允许诸如决策树和树组合之类的算法适当地处理分类特征，提高性能。
  */
object VectorIndexerExample {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val spark = SparkSession.builder().master("local").getOrCreate()
    import spark.implicits._
    val df = spark.createDataFrame(Seq(
      (0, "a","m"),
      (1, "b","w"),
      (2, "b","m"),
      (3, "a","w"),
      (4, "a","m"),
      (5, "c","w")
      )).toDF("id", "category","sex")

    val categoryIndexer = new StringIndexer().setInputCol("category").setOutputCol("categoryIndex")
    val sexIndexer = new StringIndexer().setInputCol("sex").setOutputCol("sexIndex")
    val vecDF = new VectorAssembler().setInputCols(Array("id","categoryIndex","sexIndex")).setOutputCol("features")
    val vectorIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexed").setMaxCategories(4)

    val pipeline = new Pipeline().setStages(Array(categoryIndexer,sexIndexer,vecDF,vectorIndexer))

    val model = pipeline.fit(df)

    val data = model.transform(df)
    data.show(false)
  }
}
