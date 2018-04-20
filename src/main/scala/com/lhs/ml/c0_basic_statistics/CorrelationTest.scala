package com.lhs.ml.c0_basic_statistics

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.linalg.{Matrix, Vectors}
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.sql.{Row, SparkSession}

object CorrelationTest {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val spark = SparkSession.builder().master("local").getOrCreate()
    import spark.implicits._

    val data = Seq(
      Vectors.sparse(5, Seq((0, 1.0), (3, -2.0))),// 一共4列，（index=0，value=1.0），（index=3，value=-2.0）
      Vectors.dense(4.0, 5.0, 0.0, 3.0, 2.0),
      Vectors.dense(6.0, 7.0, 0.0, 8.0, 2.0),
      Vectors.sparse(5, Seq((0, 9.0), (3, 1.0))),// 一共4列，（index=0，value=9.0），（index=3，value=1.0）
      Vectors.sparse(5, Seq())// 一共4列,都是0
    )
    data.foreach(v=>println(v))
    /**
      * (4,[0,3],[1.0,-2.0])  一共4列，（index=0，value=1.0），（index=3，value=-2.0）
      * [4.0,5.0,0.0,3.0]
      * [6.0,7.0,0.0,8.0]
      * (4,[0,3],[9.0,1.0]) 一共4列，（index=0，value=9.0），（index=3，value=1.0）
      * (4,[],[]) 一共4列,都是0
      */
    data.foreach(v=>println(v.toDense))
    /**
      * [1.0,0.0,0.0,-2.0]
      * [4.0,5.0,0.0,3.0]
      * [6.0,7.0,0.0,8.0]
      * [9.0,0.0,0.0,1.0]
      * [0.0,0.0,0.0,0.0]
      */

    val df = data.map(Tuple1.apply(_)).toDF("features")

    val Row(coeff1:Matrix) = Correlation.corr(df,"features").head()

    println(s"Pearson correlation matrix:\n $coeff1")

    val Row(coeff2: Matrix) = Correlation.corr(df, "features", "spearman").head
    println(s"Spearman correlation matrix:\n $coeff2")

  }
}
