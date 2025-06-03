from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import *


def create_optimized_job():
    w = WorkspaceClient()

    # Job cluster configuration with autoscaling
    job_cluster = JobCluster(
        job_cluster_key="historian-cluster",
        new_cluster=ClusterSpec(
            cluster_name="historian-processing",
            spark_version="13.3.x-scala2.12",
            node_type_id="i3.xlarge",
            driver_node_type_id="i3.xlarge",
            autoscale=AutoScale(min_workers=1, max_workers=10),
            enable_elastic_disk=True,
            runtime_engine=RuntimeEngine.PHOTON,  # Enable Photon for optimization
            spark_conf={
                "spark.databricks.delta.preview.enabled": "true",
                "spark.sql.adaptive.enabled": "true",
                "spark.sql.adaptive.coalescePartitions.enabled": "true",
            },
        ),
    )

    # Define job tasks
    job_tasks = [
        Task(
            task_key="ingest_bronze",
            job_cluster_key="historian-cluster",
            notebook_task=NotebookTask(
                notebook_path="/ingestion/bronze_layer",
                base_parameters={"env": "prod"},
            ),
        ),
        Task(
            task_key="process_silver",
            depends_on=[TaskDependency(task_key="ingest_bronze")],
            job_cluster_key="historian-cluster",
            notebook_task=NotebookTask(
                notebook_path="/processing/silver_layer"
            ),
        ),
        Task(
            task_key="create_gold",
            depends_on=[TaskDependency(task_key="process_silver")],
            job_cluster_key="historian-cluster",
            notebook_task=NotebookTask(notebook_path="/processing/gold_layer"),
        ),
        Task(
            task_key="ml_training",
            depends_on=[TaskDependency(task_key="create_gold")],
            job_cluster_key="historian-cluster",
            notebook_task=NotebookTask(notebook_path="/ml/training_pipeline"),
        ),
    ]

    # Create job with scheduling
    job = w.jobs.create(
        name="historian-data-pipeline",
        job_clusters=[job_cluster],
        tasks=job_tasks,
        schedule=CronSchedule(
            cron_expression="0 2 * * *",  # Daily at 2 AM
            timezone_id="UTC",
        ),
        timeout_seconds=7200,  # 2 hours timeout
        max_concurrent_runs=1,
    )

    return job


# Delta cache optimization
def optimize_delta_tables():
    spark = SparkSession.builder.getOrCreate()

    # Cache frequently accessed tables
    spark.sql("CACHE SELECT * FROM historian.core_metrics")
    spark.sql(
        "CACHE SELECT * FROM historian.gold_features WHERE timestamp >= current_date() - interval 7 days"
    )

    # Optimize Delta tables
    spark.sql("OPTIMIZE historian.silver_data ZORDER BY (tag_id, timestamp)")
    spark.sql(
        "OPTIMIZE historian.gold_features ZORDER BY (data_category, timestamp)"
    )

    # Vacuum old files (retain 7 days)
    spark.sql("VACUUM historian.silver_data RETAIN 168 HOURS")
    spark.sql("VACUUM historian.gold_features RETAIN 168 HOURS")
