from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window


def create_gold_features():
    spark = SparkSession.builder.getOrCreate()

    # Read from Silver layer
    silver_df = spark.table("historian.silver_data")

    # Feature engineering for ML consumption
    window_spec = (
        Window.partitionBy("tag_id").orderBy("timestamp").rowsBetween(-5, 0)
    )

    gold_df = (
        silver_df.withColumn("value_ma_5", avg("value").over(window_spec))
        .withColumn("value_std_5", stddev("value").over(window_spec))
        .withColumn("value_lag_1", lag("value", 1).over(window_spec))
        .withColumn("hour_of_day", hour("timestamp"))
        .withColumn("day_of_week", dayofweek("timestamp"))
        .filter(col("value_ma_5").isNotNull())
    )

    # Write to Gold Delta table
    (
        gold_df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .partitionBy("data_category", "hour_of_day")
        .saveAsTable("historian.gold_features")
    )


# Common core data processing (20% of data)
def process_core_data():
    spark = SparkSession.builder.getOrCreate()

    # Process common metrics used across multiple pipelines
    core_metrics = (
        spark.table("historian.silver_data")
        .filter(col("data_category").isin(["temperature", "pressure"]))
        .groupBy("tag_id", window("timestamp", "1 hour"))
        .agg(
            avg("value").alias("avg_value"),
            max("value").alias("max_value"),
            min("value").alias("min_value"),
            count("*").alias("record_count"),
        )
    )

    # Cache frequently accessed data
    core_metrics.cache()

    # Write to core metrics table
    (
        core_metrics.write.format("delta")
        .mode("overwrite")
        .saveAsTable("historian.core_metrics")
    )
