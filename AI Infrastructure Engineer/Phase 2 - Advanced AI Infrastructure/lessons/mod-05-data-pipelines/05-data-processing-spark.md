# Lesson 05: Data Processing with Apache Spark

## Overview
Apache Spark is a unified analytics engine for large-scale data processing. In ML infrastructure, Spark is essential for preprocessing training data, feature engineering at scale, and batch inference. This lesson covers Spark fundamentals and integration with ML pipelines.

**Duration:** 8-10 hours
**Difficulty:** Intermediate
**Prerequisites:** Python, basic SQL, understanding of distributed systems

## Learning Objectives
By the end of this lesson, you will be able to:
- Understand Spark architecture and execution model
- Use PySpark for data processing and transformation
- Implement efficient data pipelines with DataFrames
- Optimize Spark jobs for ML workloads
- Integrate Spark with ML training pipelines
- Debug and troubleshoot Spark applications

---

## 1. Introduction to Apache Spark

### 1.1 What is Apache Spark?

Apache Spark is a multi-language engine for executing data engineering, data science, and machine learning on single-node machines or clusters. It provides:

- **Speed:** In-memory processing (100x faster than MapReduce)
- **Ease of use:** High-level APIs in Python, Scala, Java, R
- **Unified engine:** Batch processing, streaming, SQL, ML, graph processing
- **Scalability:** From laptops to thousands of nodes

### 1.2 Why Spark for ML Infrastructure?

```python
# Traditional approach: Limited by single machine memory
import pandas as pd
df = pd.read_csv("training_data.csv")  # Fails if data > RAM

# Spark approach: Distributed processing
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("ML Data Prep").getOrCreate()
df = spark.read.csv("training_data.csv", header=True)  # Scales to TBs
```

**Use cases in AI infrastructure:**
- Large-scale feature engineering (millions of features)
- Data preprocessing for model training
- Batch inference on large datasets
- ETL pipelines for ML data lakes
- Feature store computation

### 1.3 Spark Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Driver Program                       │
│  ┌──────────────────────────────────────────────────┐  │
│  │           SparkContext / SparkSession            │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   Worker 1   │ │   Worker 2   │ │   Worker N   │
│  ┌────────┐  │ │  ┌────────┐  │ │  ┌────────┐  │
│  │Executor│  │ │  │Executor│  │ │  │Executor│  │
│  │ Cache  │  │ │  │ Cache  │  │ │  │ Cache  │  │
│  │ Task   │  │ │  │ Task   │  │ │  │ Task   │  │
│  └────────┘  │ │  └────────┘  │ │  └────────┘  │
└──────────────┘ └──────────────┘ └──────────────┘
```

**Key components:**
- **Driver:** Orchestrates job execution, maintains metadata
- **Executors:** Distributed workers that execute tasks
- **Cluster Manager:** Allocates resources (YARN, Kubernetes, Standalone)
- **RDD/DataFrame:** Distributed data abstractions

---

## 2. Setting Up Spark for ML Workloads

### 2.1 Installation and Configuration

```bash
# Install PySpark
pip install pyspark==3.5.0 pyarrow pandas

# Set up environment variables
export SPARK_HOME=/opt/spark
export PYSPARK_PYTHON=python3
export PYSPARK_DRIVER_PYTHON=python3
```

### 2.2 Creating a SparkSession

```python
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf

# Configure Spark for ML workloads
conf = SparkConf() \
    .setAppName("ML Data Processing") \
    .set("spark.executor.memory", "4g") \
    .set("spark.driver.memory", "2g") \
    .set("spark.executor.cores", "2") \
    .set("spark.sql.shuffle.partitions", "200") \
    .set("spark.default.parallelism", "100") \
    .set("spark.sql.adaptive.enabled", "true")

# Create SparkSession
spark = SparkSession.builder \
    .config(conf=conf) \
    .getOrCreate()

print(f"Spark version: {spark.version}")
print(f"Spark UI available at: {spark.sparkContext.uiWebUrl}")
```

### 2.3 Understanding Configuration Parameters

| Parameter | Purpose | Recommended Value |
|-----------|---------|-------------------|
| `spark.executor.memory` | Memory per executor | 4-8GB for ML |
| `spark.executor.cores` | Cores per executor | 2-5 (balance parallelism) |
| `spark.sql.shuffle.partitions` | Shuffle parallelism | 2-3x cores |
| `spark.sql.adaptive.enabled` | Dynamic optimization | true |
| `spark.serializer` | Object serialization | KryoSerializer |

```python
# Optimized configuration for ML workloads
ml_optimized_conf = SparkConf() \
    .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .set("spark.kryoserializer.buffer.max", "512m") \
    .set("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .set("spark.sql.adaptive.coalescePartitions.enabled", "true")
```

---

## 3. PySpark DataFrames and SQL

### 3.1 Creating DataFrames

```python
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType

# Method 1: From Python list
data = [
    Row(user_id=1, age=25, country="US", spend=150.0),
    Row(user_id=2, age=30, country="UK", spend=200.0),
    Row(user_id=3, age=22, country="US", spend=120.0)
]
df = spark.createDataFrame(data)

# Method 2: From CSV with schema inference
df_csv = spark.read.csv(
    "s3://ml-data/training_data.csv",
    header=True,
    inferSchema=True
)

# Method 3: With explicit schema (recommended for production)
schema = StructType([
    StructField("user_id", IntegerType(), nullable=False),
    StructField("age", IntegerType(), nullable=True),
    StructField("country", StringType(), nullable=True),
    StructField("spend", FloatType(), nullable=True)
])

df_with_schema = spark.read.csv(
    "s3://ml-data/training_data.csv",
    header=True,
    schema=schema
)

# Method 4: From Parquet (most efficient for ML)
df_parquet = spark.read.parquet("s3://ml-data/features/*.parquet")

# Show DataFrame info
df.printSchema()
df.show(5)
print(f"Total rows: {df.count()}")
```

### 3.2 DataFrame Operations

```python
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# SELECT: Column selection
selected_df = df.select("user_id", "age", "spend")

# FILTER: Row filtering
filtered_df = df.filter(
    (F.col("age") >= 25) &
    (F.col("country") == "US")
)

# WITH COLUMN: Add/modify columns
enriched_df = df \
    .withColumn("age_group",
                F.when(F.col("age") < 25, "young")
                 .when(F.col("age") < 40, "adult")
                 .otherwise("senior")) \
    .withColumn("spend_category",
                F.when(F.col("spend") < 100, "low")
                 .when(F.col("spend") < 200, "medium")
                 .otherwise("high"))

# AGGREGATIONS
agg_df = df.groupBy("country") \
    .agg(
        F.count("*").alias("user_count"),
        F.avg("spend").alias("avg_spend"),
        F.max("spend").alias("max_spend"),
        F.stddev("spend").alias("spend_stddev")
    )

# WINDOW FUNCTIONS: Useful for feature engineering
window_spec = Window.partitionBy("country").orderBy(F.desc("spend"))

ranked_df = df.withColumn(
    "spend_rank",
    F.row_number().over(window_spec)
).withColumn(
    "running_total",
    F.sum("spend").over(window_spec.rowsBetween(Window.unboundedPreceding, 0))
)

# JOINS: Combining datasets
user_profiles = spark.read.parquet("s3://ml-data/user_profiles.parquet")
transactions = spark.read.parquet("s3://ml-data/transactions.parquet")

# Inner join
combined_df = user_profiles.join(
    transactions,
    on="user_id",
    how="inner"
)

# Broadcast join (for small dimension tables)
from pyspark.sql.functions import broadcast

country_mapping = spark.read.csv("country_codes.csv", header=True)
enriched_df = df.join(
    broadcast(country_mapping),
    on="country",
    how="left"
)
```

### 3.3 SQL Interface

```python
# Register DataFrame as temporary view
df.createOrReplaceTempView("users")

# Execute SQL queries
sql_result = spark.sql("""
    SELECT
        country,
        age_group,
        COUNT(*) as user_count,
        AVG(spend) as avg_spend,
        PERCENTILE_APPROX(spend, 0.5) as median_spend,
        PERCENTILE_APPROX(spend, 0.95) as p95_spend
    FROM (
        SELECT
            *,
            CASE
                WHEN age < 25 THEN 'young'
                WHEN age < 40 THEN 'adult'
                ELSE 'senior'
            END as age_group
        FROM users
    )
    GROUP BY country, age_group
    HAVING user_count > 10
    ORDER BY country, avg_spend DESC
""")

sql_result.show()
```

---

## 4. Feature Engineering with Spark

### 4.1 Numerical Features

```python
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Statistical transformations
features_df = df \
    .withColumn("spend_log", F.log1p(F.col("spend"))) \
    .withColumn("spend_sqrt", F.sqrt(F.col("spend"))) \
    .withColumn("spend_squared", F.pow(F.col("spend"), 2))

# Scaling (manual approach)
spend_stats = df.agg(
    F.mean("spend").alias("spend_mean"),
    F.stddev("spend").alias("spend_std")
).collect()[0]

normalized_df = df.withColumn(
    "spend_normalized",
    (F.col("spend") - spend_stats.spend_mean) / spend_stats.spend_std
)

# Rolling aggregations
window_7d = Window.partitionBy("user_id") \
    .orderBy("date") \
    .rowsBetween(-6, 0)

df_with_rolling = df \
    .withColumn("spend_7d_avg", F.avg("spend").over(window_7d)) \
    .withColumn("spend_7d_sum", F.sum("spend").over(window_7d)) \
    .withColumn("spend_7d_max", F.max("spend").over(window_7d))

# Percentile ranks for outlier detection
df_with_percentiles = df.withColumn(
    "spend_percentile",
    F.percent_rank().over(Window.orderBy("spend"))
)
```

### 4.2 Categorical Features

```python
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline

# String indexing
indexer = StringIndexer(
    inputCol="country",
    outputCol="country_index",
    handleInvalid="keep"
)

# One-hot encoding
encoder = OneHotEncoder(
    inputCols=["country_index"],
    outputCols=["country_onehot"]
)

# Create pipeline
pipeline = Pipeline(stages=[indexer, encoder])
model = pipeline.fit(df)
encoded_df = model.transform(df)

# Manual one-hot with SQL
df.createOrReplaceTempView("data")
onehot_df = spark.sql("""
    SELECT
        *,
        CASE WHEN country = 'US' THEN 1 ELSE 0 END as country_us,
        CASE WHEN country = 'UK' THEN 1 ELSE 0 END as country_uk,
        CASE WHEN country = 'CA' THEN 1 ELSE 0 END as country_ca
    FROM data
""")

# Frequency encoding
country_freq = df.groupBy("country").count()
freq_encoded_df = df.join(
    country_freq.withColumnRenamed("count", "country_freq"),
    on="country"
)
```

### 4.3 Temporal Features

```python
from pyspark.sql.functions import to_timestamp, date_format, dayofweek, hour

# Parse timestamps
temporal_df = df \
    .withColumn("timestamp", to_timestamp("event_time", "yyyy-MM-dd HH:mm:ss")) \
    .withColumn("date", F.to_date("timestamp")) \
    .withColumn("year", F.year("timestamp")) \
    .withColumn("month", F.month("timestamp")) \
    .withColumn("day", F.dayofmonth("timestamp")) \
    .withColumn("hour", F.hour("timestamp")) \
    .withColumn("day_of_week", F.dayofweek("timestamp")) \
    .withColumn("is_weekend",
                F.when(F.dayofweek("timestamp").isin([1, 7]), 1)
                 .otherwise(0))

# Time-based aggregations for ML
user_temporal_features = temporal_df.groupBy("user_id") \
    .agg(
        F.countDistinct("date").alias("active_days"),
        F.min("timestamp").alias("first_seen"),
        F.max("timestamp").alias("last_seen"),
        F.avg("hour").alias("avg_activity_hour")
    ) \
    .withColumn("days_since_first",
                F.datediff(F.current_date(), F.col("first_seen"))) \
    .withColumn("days_since_last",
                F.datediff(F.current_date(), F.col("last_seen")))
```

### 4.4 Text Features

```python
from pyspark.ml.feature import Tokenizer, HashingTF, IDF

# Tokenization
tokenizer = Tokenizer(inputCol="description", outputCol="words")
words_df = tokenizer.transform(df)

# TF-IDF
hashingTF = HashingTF(inputCol="words", outputCol="raw_features", numFeatures=1000)
tf_df = hashingTF.transform(words_df)

idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
idf_model = idf.fit(tf_df)
tfidf_df = idf_model.transform(tf_df)

# Simple text statistics
text_stats_df = df \
    .withColumn("text_length", F.length("description")) \
    .withColumn("word_count", F.size(F.split("description", " "))) \
    .withColumn("contains_urgent",
                F.when(F.lower("description").contains("urgent"), 1)
                 .otherwise(0))
```

---

## 5. ML Pipeline Integration

### 5.1 Preparing Data for Model Training

```python
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors

# Assemble features into a single vector column
feature_columns = [
    "age", "spend", "spend_log", "spend_7d_avg",
    "country_index", "spend_percentile", "active_days"
]

assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features",
    handleInvalid="skip"
)

# Create feature vector
ml_ready_df = assembler.transform(df)

# Select only necessary columns for training
training_df = ml_ready_df.select("features", "label")

# Split data
train_df, test_df = training_df.randomSplit([0.8, 0.2], seed=42)

print(f"Training samples: {train_df.count()}")
print(f"Test samples: {test_df.count()}")
```

### 5.2 Feature Store Pattern

```python
from datetime import datetime

class SparkFeatureStore:
    """Simple feature store implementation with Spark"""

    def __init__(self, base_path: str):
        self.base_path = base_path

    def compute_features(self, raw_df):
        """Compute features from raw data"""
        features_df = raw_df \
            .withColumn("spend_log", F.log1p(F.col("spend"))) \
            .withColumn("age_normalized",
                       (F.col("age") - 30) / 15) \
            .withColumn("created_at", F.current_timestamp())

        return features_df

    def write_features(self, features_df, feature_group: str):
        """Write features to storage (versioned)"""
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{self.base_path}/{feature_group}/v_{version}"

        features_df.write \
            .mode("overwrite") \
            .partitionBy("country") \
            .parquet(output_path)

        # Update latest pointer
        latest_path = f"{self.base_path}/{feature_group}/latest"
        features_df.write \
            .mode("overwrite") \
            .partitionBy("country") \
            .parquet(latest_path)

        return output_path

    def read_features(self, feature_group: str, version: str = "latest"):
        """Read features from storage"""
        path = f"{self.base_path}/{feature_group}/{version}"
        return spark.read.parquet(path)

# Usage
feature_store = SparkFeatureStore("s3://ml-features")
features = feature_store.compute_features(raw_df)
version_path = feature_store.write_features(features, "user_features")
```

### 5.3 Data Quality Checks

```python
def validate_ml_data(df, config):
    """
    Validate data quality for ML training

    Args:
        df: Input DataFrame
        config: Validation configuration

    Returns:
        Validation report
    """
    validation_results = {}

    # Check for null values
    null_counts = df.select([
        F.sum(F.col(c).isNull().cast("int")).alias(c)
        for c in df.columns
    ]).collect()[0]

    validation_results["null_counts"] = null_counts.asDict()

    # Check for duplicates
    total_rows = df.count()
    unique_rows = df.dropDuplicates().count()
    validation_results["duplicate_rows"] = total_rows - unique_rows

    # Check value ranges
    for col, (min_val, max_val) in config.get("ranges", {}).items():
        out_of_range = df.filter(
            (F.col(col) < min_val) | (F.col(col) > max_val)
        ).count()
        validation_results[f"{col}_out_of_range"] = out_of_range

    # Check for outliers using IQR
    for col in config.get("numeric_columns", []):
        quantiles = df.approxQuantile(col, [0.25, 0.75], 0.01)
        if len(quantiles) == 2:
            q1, q3 = quantiles
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outlier_count = df.filter(
                (F.col(col) < lower_bound) | (F.col(col) > upper_bound)
            ).count()

            validation_results[f"{col}_outliers"] = outlier_count

    return validation_results

# Usage
config = {
    "ranges": {
        "age": (0, 120),
        "spend": (0, 10000)
    },
    "numeric_columns": ["age", "spend"]
}

validation_report = validate_ml_data(df, config)
print(validation_report)
```

---

## 6. Performance Optimization

### 6.1 Partitioning Strategies

```python
# Repartition for better parallelism
df_repartitioned = df.repartition(200)

# Partition by column for efficient joins
df_partitioned = df.repartition("country")

# Coalesce to reduce partitions (no shuffle)
df_coalesced = df_repartitioned.coalesce(50)

# Check partition distribution
print(f"Number of partitions: {df.rdd.getNumPartitions()}")

# Custom partitioning for skewed data
from pyspark.sql.functions import hash

df_custom_partitioned = df.repartition(
    100,
    F.hash(F.col("user_id"))
)
```

### 6.2 Caching and Persistence

```python
# Cache frequently accessed DataFrames
df_cached = df.cache()
df_cached.count()  # Materialize cache

# Different storage levels
from pyspark import StorageLevel

df.persist(StorageLevel.MEMORY_AND_DISK)
df.persist(StorageLevel.DISK_ONLY)

# Unpersist when done
df.unpersist()

# Example: Iterative feature engineering
base_features = compute_base_features(df).cache()

for iteration in range(5):
    enhanced = add_interaction_features(base_features, iteration)
    model = train_model(enhanced)

base_features.unpersist()
```

### 6.3 Broadcast Variables

```python
# Broadcast small lookup tables
country_mapping_dict = {
    "US": "United States",
    "UK": "United Kingdom",
    "CA": "Canada"
}

broadcast_mapping = spark.sparkContext.broadcast(country_mapping_dict)

# Use in UDF
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

@udf(StringType())
def expand_country_code(code):
    return broadcast_mapping.value.get(code, "Unknown")

df_expanded = df.withColumn(
    "country_name",
    expand_country_code(F.col("country"))
)
```

### 6.4 Avoiding Common Performance Pitfalls

```python
# BAD: Collect large datasets
# large_data = df.collect()  # DON'T DO THIS

# GOOD: Sample for inspection
sample_data = df.sample(fraction=0.01).collect()

# BAD: Multiple passes over data
# count_us = df.filter(F.col("country") == "US").count()
# count_uk = df.filter(F.col("country") == "UK").count()

# GOOD: Single pass with aggregation
country_counts = df.groupBy("country").count().collect()

# BAD: UDFs (slow Python serialization)
@udf(StringType())
def slow_processing(value):
    return value.upper()

# GOOD: Built-in functions
df_fast = df.withColumn("country_upper", F.upper(F.col("country")))

# BAD: Wide transformations without partitioning
# result = df1.join(df2, on="key")

# GOOD: Repartition before expensive operations
result = df1.repartition("key").join(
    df2.repartition("key"),
    on="key"
)
```

---

## 7. Monitoring and Debugging

### 7.1 Spark UI Analysis

```python
# Access Spark UI
print(f"Spark UI: {spark.sparkContext.uiWebUrl}")

# Key metrics to monitor:
# - Job execution time
# - Stage timing
# - Task skew
# - Shuffle read/write
# - GC time
```

### 7.2 Logging and Metrics

```python
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_with_logging(df):
    """Process data with comprehensive logging"""
    start_time = time.time()

    logger.info(f"Starting processing. Input rows: {df.count()}")
    logger.info(f"Input partitions: {df.rdd.getNumPartitions()}")

    # Processing steps
    processed = df.filter(F.col("age") > 18)
    logger.info(f"After filtering: {processed.count()} rows")

    features = compute_features(processed)
    logger.info(f"Features computed: {features.columns}")

    # Execution stats
    elapsed = time.time() - start_time
    logger.info(f"Processing completed in {elapsed:.2f}s")

    return features
```

### 7.3 Query Execution Plans

```python
# Explain query execution plan
df.explain()

# More detailed explanation
df.explain(extended=True)

# Analyze specific transformations
filtered_df = df.filter(F.col("age") > 25)
aggregated_df = filtered_df.groupBy("country").count()

print("=== Physical Plan ===")
aggregated_df.explain(mode="formatted")
```

---

## 8. Real-World ML Example

### 8.1 Complete Feature Engineering Pipeline

```python
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

class MLDataProcessor:
    """Production-grade ML data processing with Spark"""

    def __init__(self, spark: SparkSession):
        self.spark = spark

    def load_raw_data(self, path: str):
        """Load and validate raw data"""
        df = self.spark.read.parquet(path)

        # Basic validation
        assert df.count() > 0, "Empty dataset"
        assert "user_id" in df.columns, "Missing user_id"

        return df

    def create_temporal_features(self, df):
        """Create time-based features"""
        window_7d = Window.partitionBy("user_id") \
            .orderBy("date") \
            .rowsBetween(-6, 0)

        window_30d = Window.partitionBy("user_id") \
            .orderBy("date") \
            .rowsBetween(-29, 0)

        return df \
            .withColumn("spend_7d_avg", F.avg("spend").over(window_7d)) \
            .withColumn("spend_30d_avg", F.avg("spend").over(window_30d)) \
            .withColumn("transaction_7d_count",
                       F.count("*").over(window_7d)) \
            .withColumn("spend_volatility",
                       F.stddev("spend").over(window_30d))

    def create_user_features(self, df):
        """Create user-level aggregations"""
        user_features = df.groupBy("user_id").agg(
            F.count("*").alias("total_transactions"),
            F.sum("spend").alias("lifetime_spend"),
            F.avg("spend").alias("avg_spend"),
            F.stddev("spend").alias("spend_std"),
            F.min("date").alias("first_transaction"),
            F.max("date").alias("last_transaction"),
            F.countDistinct("category").alias("unique_categories")
        )

        # Add derived features
        user_features = user_features \
            .withColumn("days_active",
                       F.datediff("last_transaction", "first_transaction")) \
            .withColumn("avg_daily_spend",
                       F.col("lifetime_spend") / F.greatest(F.col("days_active"), F.lit(1))) \
            .withColumn("spend_cv",
                       F.col("spend_std") / F.col("avg_spend"))

        return user_features

    def prepare_for_training(self, df, feature_cols, label_col):
        """Prepare final dataset for model training"""
        # Assemble features
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features_raw",
            handleInvalid="skip"
        )

        # Scale features
        scaler = StandardScaler(
            inputCol="features_raw",
            outputCol="features",
            withStd=True,
            withMean=True
        )

        pipeline = Pipeline(stages=[assembler, scaler])
        model = pipeline.fit(df)

        transformed = model.transform(df)
        return transformed.select("user_id", "features", label_col)

    def run_pipeline(self, input_path: str, output_path: str):
        """Execute complete pipeline"""
        # Load data
        raw_df = self.load_raw_data(input_path)

        # Create features
        temporal_features = self.create_temporal_features(raw_df)
        user_features = self.create_user_features(temporal_features)

        # Join back to get final dataset
        final_df = temporal_features.join(user_features, on="user_id")

        # Prepare for training
        feature_columns = [
            "spend_7d_avg", "spend_30d_avg", "transaction_7d_count",
            "spend_volatility", "lifetime_spend", "avg_spend",
            "unique_categories", "spend_cv"
        ]

        training_ready = self.prepare_for_training(
            final_df,
            feature_columns,
            "label"
        )

        # Write output
        training_ready.write \
            .mode("overwrite") \
            .partitionBy("date") \
            .parquet(output_path)

        return training_ready

# Execute pipeline
processor = MLDataProcessor(spark)
result = processor.run_pipeline(
    "s3://ml-data/raw/transactions",
    "s3://ml-data/processed/training_data"
)
```

---

## 9. Best Practices

### 9.1 Data Processing

✅ **DO:**
- Use explicit schemas for production pipelines
- Partition data appropriately for your access patterns
- Cache intermediate results that are reused
- Use built-in functions instead of UDFs
- Test on sample data before running on full dataset

❌ **DON'T:**
- Collect large datasets to driver
- Use wide transformations without repartitioning
- Create too many small files (< 128MB)
- Ignore data skew
- Mix batch and streaming logic unnecessarily

### 9.2 Performance

```python
# Optimize joins
df1.repartition("key").join(df2.repartition("key"), on="key")

# Use broadcast for small tables
df.join(broadcast(small_table), on="key")

# Avoid shuffles when possible
df.filter(...).select(...).withColumn(...)  # Good: narrow transformations

# Coalesce before writing
df.coalesce(10).write.parquet(path)
```

### 9.3 Resource Management

```python
# Set appropriate resources
conf = SparkConf() \
    .set("spark.executor.memory", "8g") \
    .set("spark.executor.cores", "4") \
    .set("spark.dynamicAllocation.enabled", "true") \
    .set("spark.dynamicAllocation.minExecutors", "2") \
    .set("spark.dynamicAllocation.maxExecutors", "20")
```

---

## 10. Hands-On Exercise

### Exercise: Build an ML Feature Pipeline

**Objective:** Create a complete feature engineering pipeline for a churn prediction model.

**Dataset:** User transaction logs (CSV)
```
user_id, transaction_date, amount, category, device_type
```

**Requirements:**
1. Load data from CSV with appropriate schema
2. Create temporal features (7d, 30d aggregations)
3. Create user-level features (lifetime stats)
4. Handle missing values appropriately
5. Assemble features into vectors
6. Split into train/test sets (80/20)
7. Write output to Parquet with partitioning

**Starter Code:**
```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# TODO: Create SparkSession with appropriate configuration

# TODO: Define schema for input data

# TODO: Load data with schema

# TODO: Create temporal features
# - transaction_count_7d
# - total_spend_7d
# - avg_spend_7d
# - transaction_count_30d

# TODO: Create user-level features
# - total_transactions
# - lifetime_spend
# - days_since_first_transaction
# - favorite_category

# TODO: Handle missing values

# TODO: Assemble features and split data

# TODO: Write output
```

**Solution available in:** [`ai-infra-engineer-solutions/modules/mod-105-data-pipelines/exercise-05-spark-batch-processing/`](https://github.com/ai-infra-curriculum/ai-infra-engineer-solutions/tree/main/modules/mod-105-data-pipelines/exercise-05-spark-batch-processing)

---

## 11. Additional Resources

### Documentation
- [Apache Spark Documentation](https://spark.apache.org/docs/latest/)
- [PySpark API Reference](https://spark.apache.org/docs/latest/api/python/)
- [Spark SQL Guide](https://spark.apache.org/docs/latest/sql-programming-guide.html)

### Books
- "Learning Spark, 2nd Edition" by Damji et al.
- "High Performance Spark" by Karau & Warren
- "Spark: The Definitive Guide" by Chambers & Zaharia

### Online Courses
- Databricks Academy: Apache Spark Programming
- Coursera: Big Data Analysis with Spark
- LinkedIn Learning: Spark for ML

### Tools
- Spark UI: Monitor job execution
- Ganglia: Cluster monitoring
- Databricks: Managed Spark platform

---

## Summary

In this lesson, you learned:
- ✅ Spark architecture and execution model
- ✅ PySpark DataFrame operations for data processing
- ✅ Feature engineering at scale
- ✅ ML pipeline integration
- ✅ Performance optimization techniques
- ✅ Monitoring and debugging strategies
- ✅ Production best practices

**Next Steps:**
- Complete the hands-on exercise
- Experiment with different partitioning strategies
- Profile a Spark job using the Spark UI
- Read about Spark Structured Streaming for real-time features

**Next Lesson:** [06 - Streaming Data with Apache Kafka](./06-streaming-data-kafka.md)
