from delta.tables import DeltaTable
from pyspark.sql import SparkSession


def ensure_data_consistency():
    spark = SparkSession.builder.getOrCreate()

    # Check data quality using time travel
    current_data = spark.table("historian.silver_data")
    previous_version = (
        spark.read.format("delta")
        .option("versionAsOf", 1)
        .table("historian.silver_data")
    )

    # Compare record counts
    current_count = current_data.count()
    previous_count = previous_version.count()

    if current_count < previous_count * 0.9:  # Alert if 10% drop
        print(
            f"Data quality alert: Current count {current_count} vs previous {previous_count}"
        )

    # Rollback if needed
    def rollback_to_version(table_name, version):
        spark.sql(f"RESTORE TABLE {table_name} TO VERSION AS OF {version}")


# Delta Lake optimization utilities
def delta_maintenance():
    spark = SparkSession.builder.getOrCreate()

    tables = [
        "historian.bronze_data",
        "historian.silver_data",
        "historian.gold_features",
    ]

    for table in tables:
        # Optimize file sizes
        spark.sql(f"OPTIMIZE {table}")

        # Update table statistics
        spark.sql(f"ANALYZE TABLE {table} COMPUTE STATISTICS")

        # Clean up old files
        spark.sql(f"VACUUM {table} RETAIN 168 HOURS")
