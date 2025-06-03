import mlflow
import mlflow.spark
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator


def ml_training_pipeline():
    spark = SparkSession.builder.getOrCreate()

    # Enable MLflow autologging
    mlflow.spark.autolog()

    with mlflow.start_run(run_name="historian_prediction_model"):
        # Load feature data
        features_df = spark.table("historian.gold_features")

        # Prepare features
        feature_cols = [
            "value_ma_5",
            "value_std_5",
            "value_lag_1",
            "hour_of_day",
            "day_of_week",
        ]
        assembler = VectorAssembler(
            inputCols=feature_cols, outputCol="features"
        )
        scaler = StandardScaler(
            inputCol="features", outputCol="scaled_features"
        )

        # Model
        rf = RandomForestRegressor(
            featuresCol="scaled_features", labelCol="value", numTrees=100
        )

        # Pipeline
        pipeline = Pipeline(stages=[assembler, scaler, rf])

        # Train/test split
        train_df, test_df = features_df.randomSplit([0.8, 0.2], seed=42)

        # Train model
        model = pipeline.fit(train_df)

        # Evaluate
        predictions = model.transform(test_df)
        evaluator = RegressionEvaluator(
            labelCol="value", predictionCol="prediction", metricName="rmse"
        )
        rmse = evaluator.evaluate(predictions)

        # Log metrics
        mlflow.log_metric("rmse", rmse)

        # Register model
        mlflow.spark.log_model(
            model,
            "historian_model",
            registered_model_name="historian_predictor",
        )

    return model


# Batch inference job
def batch_inference():
    spark = SparkSession.builder.getOrCreate()

    # Load registered model
    model = mlflow.spark.load_model("models:/historian_predictor/Production")

    # Load latest data
    latest_data = spark.table("historian.gold_features").filter(
        col("timestamp") >= current_date()
    )

    # Make predictions
    predictions = model.transform(latest_data)

    # Write predictions to Delta table
    (
        predictions.select("tag_id", "timestamp", "prediction", "value")
        .write.format("delta")
        .mode("append")
        .saveAsTable("historian.predictions")
    )
