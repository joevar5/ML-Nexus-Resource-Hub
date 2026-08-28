# Exercise 05: Spark Batch Processing for ML

**Duration:** 3 hours
**Difficulty:** Intermediate
**Prerequisites:** Java + PySpark installed (lab 03 of mod-05)

## Objective

Build a PySpark job that processes a real-sized dataset (1-10 GB) end-to-end: read partitioned Parquet, transform with both SQL and DataFrame APIs, join with a dimension table via broadcast join, write back as partitioned + bucketed Parquet, and tune for performance.

## Why this matters

Spark is the default tool for "more data than fits in pandas" but less than warrants a real data warehouse. Engineers who can write performant PySpark — partition correctly, choose broadcast vs shuffle joins, avoid the common cost of unnecessary shuffles — produce jobs that finish in minutes instead of hours.

## Requirements

1. Process a 1-10 GB Parquet dataset (NYC taxi or synthetic).
2. Use both **SQL** and **DataFrame** APIs.
3. Perform at least one **broadcast join** (small dimension).
4. Perform at least one **shuffle join** with explicit partitioning.
5. Write output partitioned by date, bucketed by user.
6. Tune `spark.sql.shuffle.partitions` based on data size.
7. Produce a `BENCHMARK.md` comparing runs with different tunings.

## Step-by-step

### Step 1 — Get data + setup (20 min)
```bash
mkdir spark-batch && cd spark-batch
python -m venv venv && source venv/bin/activate
pip install 'pyspark==3.5.1' pyarrow

# 6 months of NYC taxi data ≈ 4 GB
for m in 01 02 03 04 05 06; do
  curl -LfO "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-$m.parquet" \
    -o "data/tripdata_2024_$m.parquet"
done
```

### Step 2 — Skeleton job (15 min)
```python
# job.py
from pyspark.sql import SparkSession, functions as F, Window

spark = (
    SparkSession.builder
    .appName("trip-features")
    .config("spark.driver.memory", "4g")
    .config("spark.executor.memory", "4g")
    .config("spark.sql.shuffle.partitions", "16")
    .config("spark.sql.adaptive.enabled", "true")
    .getOrCreate()
)
```

### Step 3 — Read + clean (30 min)
```python
df = spark.read.parquet("data/")
print("rows:", df.count())

clean = (df
    .filter((F.col("trip_distance") > 0) & (F.col("fare_amount") > 0))
    .filter(F.col("tpep_pickup_datetime") < F.col("tpep_dropoff_datetime"))
    .withColumn("date", F.to_date("tpep_pickup_datetime"))
    .withColumn("hour", F.hour("tpep_pickup_datetime"))
    .withColumn("trip_minutes",
        (F.unix_timestamp("tpep_dropoff_datetime") - F.unix_timestamp("tpep_pickup_datetime"))/60))
```

### Step 4 — SQL API (15 min)
```python
clean.createOrReplaceTempView("trips")
agg = spark.sql("""
    SELECT date, PULocationID,
           COUNT(*) AS trips,
           AVG(fare_amount) AS avg_fare,
           AVG(trip_minutes) AS avg_min
    FROM trips
    GROUP BY date, PULocationID
""")
```

### Step 5 — Broadcast join (30 min)
Read a small zone dimension (~250 rows) and broadcast-join:
```python
zones = spark.read.csv("zones.csv", header=True, inferSchema=True)   # ~250 rows
joined = agg.join(F.broadcast(zones), agg.PULocationID == zones.LocationID)
```
Compare against an un-broadcasted version with explain plans — see the BroadcastHashJoin vs SortMergeJoin difference.

### Step 6 — Shuffle join (30 min)
Self-join trips with an offset (e.g., previous trip of the same medallion) to demonstrate a real shuffle:
```python
w = Window.partitionBy("VendorID").orderBy("tpep_pickup_datetime")
with_prev = clean.withColumn("prev_fare", F.lag("fare_amount").over(w))
```

### Step 7 — Write partitioned + bucketed (15 min)
```python
(agg
    .repartition("date")     # one file per date partition
    .write.mode("overwrite")
    .partitionBy("date")
    .bucketBy(8, "PULocationID")
    .sortBy("PULocationID")
    .saveAsTable("trip_aggregates")
)
```

### Step 8 — Tune + benchmark (45 min)
Run 3 versions: 
- Default `spark.sql.shuffle.partitions=200` (way too many for 4GB on a laptop)
- Tuned: 16
- AQE enabled
Compare wall-clock times in `BENCHMARK.md`.

## Deliverables

1. `job.py` runnable end-to-end.
2. Output Parquet partitioned by date, bucketed by PULocationID.
3. `BENCHMARK.md` with 3 tunings + observed times.
4. `EXPLAIN_NOTES.md` paste of `df.explain()` for the broadcast join.

## Validation

- [ ] Job completes in < 5 min on 4 GB input (laptop).
- [ ] Output exists with correct partitioning structure (verify with `ls data/output/date=2024-01-15/`).
- [ ] Broadcast join EXPLAIN shows `BroadcastHashJoin`.
- [ ] AQE-enabled run produces fewer output files than default.

## Stretch goals

- Run on **AWS EMR Serverless** or **Databricks** at 10× the data size; observe the same patterns.
- Profile with **Spark UI**; identify the slowest stage and explain why.
- Add a **Delta Lake** sink with merge/upsert.

## Common pitfalls

- **`spark.sql.shuffle.partitions=200` on a laptop** — Default. Produces 200 tiny files; massive overhead. Tune to ~CPU count for laptop.
- **`groupBy` followed by `count`** — Triggers a shuffle. Reads from `_metadata` first if possible.
- **`broadcast()` on a not-actually-small table** — OOM driver. Default broadcast threshold is 10 MB; respect it.
- **`partitionBy` with too many partitions** — One file per partition; if you have 10k partitions you have 10k tiny files. Coalesce first.
