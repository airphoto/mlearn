package com.lhs.ml

import java.text.SimpleDateFormat
import java.util.Date

import org.apache.avro.SchemaBuilder.ArrayBuilder
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window
import scala.collection.mutable.ArrayBuffer

/**
 * Hello world!
 *
 */
object App {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val spark = SparkSession.builder().master("local").getOrCreate()
    import spark.implicits._

    val df = Seq((1, "a"), (1, "a"), (2, "a"), (1, "b"), (2, "b"), (3, "b")).toDF("id", "category")
//    val byCategoryOrderedById = Window.partitionBy('category).orderBy('id).rangeBetween(Window.currentRow, 1)
//    df.select($"id",$"category",row_number().over(Window.partitionBy('category).orderBy($"id")).as("range")).filter($"range"<=2).show(false)

    df.map(_.mkString("\001")).show(false)
  }
}
