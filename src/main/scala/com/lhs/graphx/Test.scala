package com.lhs.graphx

import org.apache.spark.sql.SparkSession

object Test {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local").getOrCreate()
    import spark.implicits._

    val m = spark.read.textFile("")
    val mDF = m.map(r=>{
      val Array(_,changeType,assetId,pluginId,appId,_,_,_,_,changeCount) = r.split("\\|")
      (appId,pluginId,assetId,changeType,changeCount.toLong)
    }).toDF("app_id,plugin_id,asset_id,change_type,change_count".split(","):_*).filter($"app_id".isNull || $"plugin_id".isNull || $"asset_id".isNull || $"change_type".isNull || $"app_id"==="null" || $"plugin_id"==="null" || $"asset_id"==="null" || $"change_type"==="null" || $"app_id"==="NULL" || $"plugin_id"==="NULL" || $"asset_id"==="NULL" || $"change_type"==="NULL" || $"app_id"==="" || $"plugin_id"==="" || $"asset_id"==="" || $"change_type"==="")

    spark.stop()
  }
}
