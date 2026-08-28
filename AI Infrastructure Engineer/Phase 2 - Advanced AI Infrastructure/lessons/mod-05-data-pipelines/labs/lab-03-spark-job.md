# Lab 03: Run a Spark Batch Job Locally

**Duration:** 60 min  **Prerequisites:** Java 11+, Python 3.11+

## Objective
Run PySpark locally, process a meaningful dataset (NYC taxi trips sample), and write Parquet output partitioned by date.

## Steps

### 1. Install PySpark
```bash
python -m venv venv && source venv/bin/activate
pip install 'pyspark==3.5.1' pyarrow
```

### 2. Get sample data
```bash
mkdir -p data
curl -LfO https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet
mv yellow_tripdata_2024-01.parquet data/
```

### 3. Write the Spark job
```python
# job.py
from pyspark.sql import SparkSession, functions as F

spark = (
    SparkSession.builder
    .appName("taxi-trip-aggregation")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

df = spark.read.parquet("data/yellow_tripdata_2024-01.parquet")
print(f"rows: {df.count():,}")

agg = (
    df
    .withColumn("date", F.to_date("tpep_pickup_datetime"))
    .filter(F.col("date").between("2024-01-01", "2024-01-31"))
    .groupBy("date", "PULocationID")
    .agg(
        F.count("*").alias("trips"),
        F.avg("fare_amount").alias("avg_fare"),
        F.avg("trip_distance").alias("avg_distance"),
    )
)

(
    agg
    .repartition("date")
    .write.mode("overwrite")
    .partitionBy("date")
    .parquet("data/trips_by_pickup")
)

print("done")
spark.stop()
```

### 4. Run it
```bash
python job.py
ls data/trips_by_pickup/
```

### 5. Read it back to confirm
```python
import pyspark
spark = pyspark.sql.SparkSession.builder.getOrCreate()
out = spark.read.parquet("data/trips_by_pickup")
out.show(10)
out.filter("date = '2024-01-15'").orderBy(F.desc("trips")).show(5)
```

### 6. Tune for your laptop
```python
.config("spark.driver.memory", "4g")
.config("spark.sql.shuffle.partitions", "4")    # default 200, way too many for laptop
```

## Validation
- [ ] Job completes in <2 minutes.
- [ ] Output directory contains `date=2024-01-01/`, `date=2024-01-02/`, etc.
- [ ] Re-reading produces the expected aggregated rows.

## Cleanup
```bash
rm -rf data/
```

## Troubleshooting
- **`Java HotSpot Memory error`** — Lower `spark.driver.memory` or process fewer days.
- **`Permission denied: jps`** — Java isn't on PATH; `brew install openjdk@17 && export JAVA_HOME=...`.
- **Too many output files** — `repartition("date")` before `partitionBy` keeps one file per date partition.
