from pyspark.sql import SparkSession
from pyspark.sql.types import *
from delta.tables import DeltaTable


# Bronze Layer - Raw Data Ingestion with Auto Loader
def ingest_historian_data():
    spark = SparkSession.builder.appName("HistorianIngestion").getOrCreate()

    # Define schema for historian data
    historian_schema = StructType(
        [
            StructField("timestamp", TimestampType(), True),
            StructField("tag_id", StringType(), True),
            StructField("value", DoubleType(), True),
            StructField("quality", StringType(), True),
            StructField("source_system", StringType(), True),
        ]
    )

    # Auto Loader configuration for incremental ingestion
    df = (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("cloudFiles.schemaLocation", "/mnt/historian/schema")
        .option("cloudFiles.maxFilesPerTrigger", 10)
        .schema(historian_schema)
        .load("/mnt/historian/raw/")
    )

    # Write to Bronze Delta table
    query = (
        df.writeStream.format("delta")
        .outputMode("append")
        .option("checkpointLocation", "/mnt/historian/checkpoints/bronze")
        .trigger(processingTime="5 minutes")
        .table("historian.bronze_data")
    )

    return query


# Silver Layer - Data Cleansing and Schema Enforcement
def process_silver_layer():
    spark = SparkSession.builder.getOrCreate()

    # Read from Bronze layer with incremental processing
    bronze_df = spark.readStream.format("delta").table("historian.bronze_data")

    # Data cleansing and validation
    silver_df = (
        bronze_df.filter(col("quality") == "Good")
        .filter(col("value").isNotNull())
        .withColumn("processed_timestamp", current_timestamp())
        .withColumn(
            "data_category",
            when(col("tag_id").startswith("TEMP"), "temperature")
            .when(col("tag_id").startswith("PRESS"), "pressure")
            .otherwise("other"),
        )
    )

    # Write to Silver Delta table with merge logic
    def upsert_to_silver(microBatchDF, batchId):
        if DeltaTable.isDeltaTable(spark, "historian.silver_data"):
            silver_table = DeltaTable.forName(spark, "historian.silver_data")

            silver_table.alias("target").merge(
                microBatchDF.alias("source"),
                "target.tag_id = source.tag_id AND target.timestamp = source.timestamp",
            ).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()
        else:
            microBatchDF.write.format("delta").saveAsTable(
                "historian.silver_data"
            )

    query = (
        silver_df.writeStream.foreachBatch(upsert_to_silver)
        .option("checkpointLocation", "/mnt/historian/checkpoints/silver")
        .trigger(processingTime="10 minutes")
        .start()
    )

    return query
