# Module 05: Data Pipelines — Answer Key

> Detailed answer key with rationale, common mistakes, and lesson references for the [module quiz](../../../lessons/mod-05-data-pipelines/quizzes/module-quiz.md).
>
> **Academic integrity:** For self-study after attempting the quiz.

---

## Question 1
**Q:** What is the primary purpose of Apache Airflow in ML infrastructure?

**Answer:** B) To orchestrate and schedule complex data workflows

**Explanation:**
Airflow is a workflow orchestrator. It does not perform model training, serving, or storage itself — instead it coordinates tasks (extract, transform, train, evaluate, deploy) and manages their dependencies, scheduling, retries, and observability. In ML infrastructure it is the "conductor" that wires together tools like Spark, DVC, and model trainers into reproducible pipelines.

**Common Mistakes:**
- Choosing A (train models) — Airflow does not run training code itself; it triggers training tasks defined elsewhere.
- Choosing C (serve predictions) — serving is the job of a model server (e.g., TorchServe, KServe), not an orchestrator.
- Choosing D (store training data) — storage is delegated to object stores, warehouses, or DVC remotes.

**Related Material:** `lessons/mod-05-data-pipelines/02-apache-airflow-fundamentals.md`

---

## Question 2
**Q:** In Airflow, what does DAG stand for?

**Answer:** B) Directed Acyclic Graph

**Explanation:**
A DAG models the workflow as nodes (tasks) connected by directed edges (dependencies), with the strict rule that no cycles are allowed. This guarantees a deterministic execution order and a finite run — properties Airflow's scheduler relies on to plan task launches.

**Common Mistakes:**
- Picking A, C, or D — these are plausible-sounding expansions but none are real Airflow terminology. The "acyclic" property is the load-bearing constraint and is what distinguishes a DAG from a general graph.

**Related Material:** `lessons/mod-05-data-pipelines/02-apache-airflow-fundamentals.md`

---

## Question 3
**Q:** True or False: In Airflow, tasks within a DAG can have circular dependencies.

**Answer:** False

**Explanation:**
The "A" in DAG stands for *acyclic*. If task A depends on B and B depends on A, the scheduler cannot determine a starting point and Airflow will refuse to parse the DAG. Cycles are detected at DAG-parse time and raise an `AirflowDagCycleException`.

**Common Mistakes:**
- Answering True by confusing "task can be retried multiple times" with "task can depend on itself." Retries are loop-like in *time* but the dependency graph is still acyclic.

**Related Material:** `lessons/mod-05-data-pipelines/02-apache-airflow-fundamentals.md`

---

## Question 4
**Q:** Which Airflow component is responsible for executing tasks?

**Answer:** C) Executor

**Explanation:**
Airflow separates *what to run* (scheduler decisions) from *how/where to run it* (the executor). The Scheduler decides which task instances are ready, queues them, and the Executor (Local, Celery, Kubernetes, etc.) actually launches the worker process that runs the task code.

**Common Mistakes:**
- Choosing A (Scheduler) — the scheduler only *decides* what should run; it hands work off to the executor.
- Choosing B (Web Server) — provides the UI and API; never runs task code.
- Choosing D (Metadata Database) — stores DAG/run/task state; passive storage, not an executor.

**Related Material:** `lessons/mod-05-data-pipelines/02-apache-airflow-fundamentals.md`

---

## Question 5
**Q:** What is the purpose of the `depends_on_past` parameter in Airflow?

**Answer:** A) To prevent tasks from running if the previous DAG run failed

**Explanation:**
`depends_on_past=True` makes a task instance wait for the *same task* in the *previous scheduled DAG run* to have succeeded before it will run. This is critical for stateful pipelines (e.g., incremental ETL) where today's run depends on yesterday's run having completed correctly.

**Common Mistakes:**
- Choosing B (share data between tasks) — that is XCom, not `depends_on_past`.
- Choosing C (schedule tasks in the past) — backfilling is configured via `start_date` and `catchup`, not this parameter.
- Choosing D (check task dependencies) — ordinary upstream/downstream dependencies are declared with `>>` / `set_upstream`, not `depends_on_past`.

**Related Material:** `lessons/mod-05-data-pipelines/03-advanced-airflow-ml.md`

---

## Question 6
**Q:** Explain the difference between `CeleryExecutor` and `LocalExecutor` in Airflow.

**Answer:**
- **LocalExecutor** runs tasks in parallel on the *same machine* as the scheduler, using Python multiprocessing. It is bounded by that single host's CPU, memory, and process limits — appropriate for development and small/medium production deployments.
- **CeleryExecutor** distributes tasks across a *pool of remote worker machines* via a Celery message broker (Redis or RabbitMQ). It enables horizontal scale-out, fault isolation across hosts, and is appropriate for high-throughput production workloads.

**Explanation:**
The core distinction is single-host parallelism versus distributed parallelism, with a corresponding step-up in operational complexity (you must run and monitor a broker and a worker fleet for Celery). KubernetesExecutor takes this further by launching one pod per task.

**Common Mistakes:**
- Saying LocalExecutor is "single-threaded" — it is multi-process, just on one machine.
- Forgetting to mention that CeleryExecutor requires an external message broker (Redis/RabbitMQ) and worker processes — that operational overhead is the main trade-off.
- Confusing CeleryExecutor with KubernetesExecutor; both distribute work but Kubernetes provisions a fresh pod per task while Celery uses long-running workers.

**Related Material:** `lessons/mod-05-data-pipelines/02-apache-airflow-fundamentals.md`

---

## Question 7
**Q:** What problem does DVC (Data Version Control) solve?

**Answer:** A) Managing large datasets with Git

**Explanation:**
Git is optimized for source code and breaks down on large binary files (gigabytes of images, parquet, model weights). DVC layers on top of Git: it tracks a small `.dvc` pointer file in Git while pushing the actual large file to a remote backend (S3, GCS, Azure, SSH, etc.). The result is Git-style version control for arbitrarily large datasets and models.

**Common Mistakes:**
- Choosing B (tracking model performance) — that is the job of MLflow / Weights & Biases.
- Choosing C (deploying to production) — DVC pipelines run experiments; deployment is handled by CI/CD or serving platforms.
- Choosing D (monitoring data quality) — that is Great Expectations / Deequ / Soda, not DVC.

**Related Material:** `lessons/mod-05-data-pipelines/04-data-versioning-dvc.md`

---

## Question 8
**Q:** True or False: DVC stores the actual data files in Git repositories.

**Answer:** False

**Explanation:**
DVC commits only a tiny metadata file (the `.dvc` file) containing a content hash and remote location. The actual bytes live in DVC's local cache and on the configured remote storage (S3, GCS, Azure Blob, SSH, etc.). This keeps the Git repo small and fast while still allowing reproducible checkout of any historical dataset version.

**Common Mistakes:**
- Answering True after reading "DVC works with Git" — it works *alongside* Git, not by stuffing binaries into the Git object database.

**Related Material:** `lessons/mod-05-data-pipelines/04-data-versioning-dvc.md`

---

## Question 9
**Q:** Which command is used to track a new dataset with DVC?

**Answer:** A) `dvc add <file>`

**Explanation:**
`dvc add` computes the hash of the file, moves it into the DVC cache, creates a `<file>.dvc` pointer file, and adds the original path to `.gitignore`. You then `git add` the `.dvc` file and `dvc push` to upload the data to the configured remote.

**Common Mistakes:**
- Choosing B (`dvc track`) — not a DVC command; people invent it by analogy with `git lfs track`.
- Choosing C (`dvc commit`) — exists but is for re-recording outputs of pipeline stages after manual changes, not for first-time tracking.
- Choosing D (`dvc push`) — uploads already-tracked files to the remote; it cannot start tracking a new file.

**Related Material:** `lessons/mod-05-data-pipelines/04-data-versioning-dvc.md`

---

## Question 10
**Q:** What is the purpose of a DVC pipeline?

**Answer:** B) To define reproducible data processing workflows

**Explanation:**
A DVC pipeline (defined in `dvc.yaml`) declares stages with explicit inputs (deps), outputs (outs), parameters, and the command to run. DVC tracks the hashes of each, so when you run `dvc repro` it only re-executes stages whose inputs changed — giving you reproducible, cache-aware data and ML workflows directly tied to versioned data.

**Common Mistakes:**
- Choosing A (clean data) — cleaning is something a stage might *do*; pipelines are the wrapper, not the cleaning tool.
- Choosing C (compress files) or D (encrypt) — DVC neither compresses nor encrypts data; those concerns are delegated to the remote storage backend.

**Related Material:** `lessons/mod-05-data-pipelines/04-data-versioning-dvc.md`

---

## Question 11
**Q:** What is the main advantage of Apache Spark over traditional MapReduce?

**Answer:** B) In-memory processing for 100x speedup

**Explanation:**
Hadoop MapReduce writes intermediate results to disk between every map and reduce stage. Spark keeps datasets (RDDs / DataFrames) in memory across stages via its DAG execution engine, eliminating most disk I/O. For iterative workloads (ML training, graph algorithms), this yields the well-known order-of-magnitude speedup.

**Common Mistakes:**
- Choosing A (security) — Spark and MapReduce both rely on Hadoop ecosystem security (Kerberos, Ranger); not a differentiator.
- Choosing C (smaller storage) — Spark actually uses *more* RAM by design.
- Choosing D (easier installation) — both are non-trivial to operate; this is not the headline advantage.

**Related Material:** `lessons/mod-05-data-pipelines/05-data-processing-spark.md`

---

## Question 12
**Q:** In Spark, what is a DataFrame?

**Answer:** A) A distributed collection of data organized into named columns

**Explanation:**
A Spark DataFrame is a distributed, partitioned collection of `Row` objects with a schema (named, typed columns). It is conceptually like a relational table or a pandas DataFrame but distributed across executors and optimized by Catalyst (query planner) and Tungsten (execution engine).

**Common Mistakes:**
- Choosing B (single-machine pandas DataFrame) — pandas is single-process; Spark DataFrames are distributed. They have a similar *API*, not the same execution model.
- Choosing C (SQL table) — you can query DataFrames with SQL, but the DataFrame itself is an in-memory distributed structure, not a persisted table.
- Choosing D (JSON file format) — JSON is a *source* you can read into a DataFrame, not the DataFrame itself.

**Related Material:** `lessons/mod-05-data-pipelines/05-data-processing-spark.md`

---

## Question 13
**Q:** True or False: In Spark, transformations are executed immediately when called.

**Answer:** False

**Explanation:**
Spark uses *lazy evaluation*. Transformations like `filter`, `map`, `select`, and `groupBy` only build up a logical plan (a lineage DAG). Nothing actually runs until an *action* (e.g., `count`, `collect`, `write`, `show`) triggers execution. This lets Catalyst optimize the entire plan (predicate pushdown, column pruning, stage fusion) before any data moves.

**Common Mistakes:**
- Answering True by analogy with pandas — pandas is eager; Spark is lazy. Conflating the two is the most common pitfall when moving from pandas to Spark.

**Related Material:** `lessons/mod-05-data-pipelines/05-data-processing-spark.md`

---

## Question 14
**Q:** Which of the following is a Spark action (not a transformation)?

**Answer:** D) `count()`

**Explanation:**
Actions return a value to the driver or write data out, triggering execution of the lineage. `count()` returns a `Long` to the driver and therefore must run the plan. `filter`, `map`, and `groupBy` only build the plan and return a new DataFrame/RDD — they are transformations.

**Common Mistakes:**
- Choosing C (`groupBy`) — it *looks* heavy (and does trigger a shuffle when an action runs) but by itself returns a `GroupedData` object lazily; nothing executes until you call an aggregate + action.
- Choosing A or B (`filter` / `map`) — both are classic lazy transformations.

**Related Material:** `lessons/mod-05-data-pipelines/05-data-processing-spark.md`

---

## Question 15
**Q:** Explain what "shuffle" means in Spark and why it's expensive.

**Answer:**
A shuffle is the process of *redistributing data across partitions and executors* so that rows sharing a key end up on the same partition. It is triggered by wide transformations such as `groupByKey`, `reduceByKey`, `join`, `distinct`, and `repartition`.

**Explanation:**
Shuffles are expensive because they involve:
1. **Disk I/O** — map-side outputs are written to local disk as shuffle files.
2. **Network transfer** — reducers fetch their partitions across the cluster.
3. **Serialization / deserialization** of every shuffled record.
4. **Memory pressure** — partitions that don't fit in executor memory spill to disk.
5. **Stage boundary** — a shuffle ends one stage and starts another, blocking progress until all map tasks finish.

Mitigations include using `reduceByKey` over `groupByKey`, broadcast joins for small tables, salting skewed keys, and pre-partitioning data.

**Common Mistakes:**
- Calling it "moving data between machines" without mentioning disk spill or stage boundaries — both are major contributors to cost.
- Confusing shuffle with `repartition` only — `repartition` *causes* a shuffle, but joins, group-bys, and distinct also do.
- Forgetting that `reduceByKey` shuffles *less* data than `groupByKey` because it combines map-side; this is a standard interview point.

**Related Material:** `lessons/mod-05-data-pipelines/05-data-processing-spark.md`

---

## Question 16
**Q:** What is Apache Kafka primarily used for?

**Answer:** B) Real-time event streaming

**Explanation:**
Kafka is a distributed, log-structured event streaming platform. Producers append events to topic partitions and consumers read from them, with messages retained on disk for a configurable period. It is the de facto backbone for event-driven architectures, real-time analytics, CDC, and ML feature streaming.

**Common Mistakes:**
- Choosing A (batch processing) — that is Spark/Hadoop territory; Kafka is for streaming, though Kafka can *feed* batch jobs.
- Choosing C (model training) — Kafka transports data; training happens in a framework like PyTorch / TensorFlow.
- Choosing D (data warehousing) — warehouses (Snowflake, BigQuery, Redshift) serve analytical queries; Kafka is a log, not a queryable store.

**Related Material:** `lessons/mod-05-data-pipelines/06-streaming-data-kafka.md`

---

## Question 17
**Q:** In Kafka, what is a topic?

**Answer:** B) A category or feed name to which messages are published

**Explanation:**
A topic is a named, append-only log to which producers publish records and from which consumers read. Topics are split into partitions for parallelism and replicated across brokers for durability. The topic name is the unit of subscription and access control.

**Common Mistakes:**
- Choosing A (database table) — topics are append-only streams, not mutable tables (though Kafka Streams provides a table abstraction on top).
- Choosing C (consumer instance) — a consumer *reads from* topics; it is not itself a topic.
- Choosing D (configuration file) — topic *configuration* exists, but the topic itself is the logical stream.

**Related Material:** `lessons/mod-05-data-pipelines/06-streaming-data-kafka.md`

---

## Question 18
**Q:** True or False: In Kafka, consumers in the same consumer group will receive duplicate messages.

**Answer:** False

**Explanation:**
Within a consumer group, each *partition* is assigned to exactly one consumer at a time, so each message is delivered to exactly one consumer in the group — this is how Kafka achieves horizontal scaling within a group. Different consumer *groups* reading the same topic each get their own copy of every message (broadcast across groups, load-balanced within a group).

**Common Mistakes:**
- Answering True by confusing within-group (load balanced) and across-group (broadcast) semantics.
- Forgetting that at-least-once delivery can produce duplicates on rebalance/retry — but that is a delivery-semantics edge case, not the design intent the question asks about.

**Related Material:** `lessons/mod-05-data-pipelines/06-streaming-data-kafka.md`

---

## Question 19
**Q:** What is the purpose of partitions in Kafka?

**Answer:** B) To enable parallel processing and scalability

**Explanation:**
Partitions are how Kafka achieves horizontal scale. A topic is split into N partitions; producers spread writes across them (often by key hash), and consumers in a group split partitions among themselves so each consumer reads a disjoint subset in parallel. Throughput and consumer parallelism are both bounded by partition count.

**Common Mistakes:**
- Choosing A (encrypt messages) — encryption is handled by TLS in flight and at-rest encryption on disk, not partitioning.
- Choosing C (compress data) — compression is a producer-side codec setting (gzip, snappy, lz4, zstd).
- Choosing D (validate format) — schema validation is the job of Schema Registry, not partitions.

**Related Material:** `lessons/mod-05-data-pipelines/06-streaming-data-kafka.md`

---

## Question 20
**Q:** Which of the following is NOT a dimension of data quality?

**Answer:** C) Velocity

**Explanation:**
Standard data-quality dimensions include completeness, accuracy, consistency, timeliness, validity, and uniqueness. Velocity is one of the "3 Vs of Big Data" (Volume, Velocity, Variety), which describe data *characteristics*, not data *quality*. Confusing the two frameworks is the trap.

**Common Mistakes:**
- Choosing A (Completeness) — a core quality dimension (are required fields populated?).
- Choosing B (Accuracy) — a core quality dimension (do the values match reality?).
- Choosing D (Consistency) — a core quality dimension (does the same fact agree across systems?).

**Related Material:** `lessons/mod-05-data-pipelines/07-data-quality-validation.md`

---

## Question 21
**Q:** What is data drift in the context of ML?

**Answer:** B) Statistical properties of data changing over time

**Explanation:**
Data drift (covariate / feature drift) means the input distribution P(X) seen in production has diverged from the training distribution. Concept drift is the related case where P(Y|X) changes. Both silently degrade model performance even when the pipeline runs error-free, which is why production ML systems monitor distributions, not just up-time.

**Common Mistakes:**
- Choosing A (corruption in transmission) — that is a data *quality* issue (bit errors, dropped fields), not drift.
- Choosing C (accidental deletion) — that is a data loss / governance issue.
- Choosing D (stored in wrong location) — that is a data engineering bug, not drift.

**Related Material:** `lessons/mod-05-data-pipelines/07-data-quality-validation.md`

---

## Question 22
**Q:** Name three common data quality checks you should implement in an ML training pipeline.

**Answer:** Any three of:
1. **Null / missing-value checks** — fail or alert when required columns exceed a null-rate threshold.
2. **Range / domain validation** — numeric values within bounds (e.g., `0 <= age <= 120`), categorical values in an allowed set.
3. **Schema validation** — expected columns exist with the expected types; new or missing columns trigger a hard fail.
4. **Uniqueness / primary-key checks** — ID columns are unique, no duplicate rows.
5. **Distribution / drift checks** — population stability index, KS test, or mean/std comparison vs. a reference dataset.
6. **Referential integrity** — foreign-key values exist in the referenced table.
7. **Row-count / freshness checks** — today's batch is within an expected size window and not stale.

**Explanation:**
A robust ML pipeline layers structural checks (schema, types), semantic checks (ranges, uniqueness), and statistical checks (distributions, drift). Tools such as Great Expectations, Deequ, Soda, and Pandera codify these as testable expectations that fail the pipeline before bad data reaches training.

**Common Mistakes:**
- Listing only null checks — that catches missing data but misses out-of-range values, schema drift, and distribution shift.
- Conflating model-quality monitoring (accuracy, AUC) with *data*-quality checks; the question asks about the data, not the model.
- Forgetting freshness — pipelines often run on stale data without anyone noticing until predictions degrade.

**Related Material:** `lessons/mod-05-data-pipelines/07-data-quality-validation.md`

---

## Question 23
**Q:** What are the three pillars of observability?

**Answer:** B) Metrics, Logs, Traces

**Explanation:**
The canonical observability pillars are **metrics** (numeric time-series, e.g., latency p95, request rate), **logs** (timestamped, structured event records), and **traces** (causal request paths across services). Dashboards and alerts are *consumers* of these pillars, not pillars themselves.

**Common Mistakes:**
- Choosing A (Metrics, Logs, Dashboards) — dashboards visualize metrics; they are an output, not a data type.
- Choosing C (Alerts, Logs, Reports) — alerts and reports are also outputs derived from the pillars.
- Choosing D (Metrics, Alerts, Traces) — alerts are derived from metrics/logs; logs are the missing pillar here.

**Related Material:** `lessons/mod-05-data-pipelines/08-pipeline-monitoring-errors.md`

---

## Question 24
**Q:** What is a Dead Letter Queue (DLQ)?

**Answer:** B) A storage location for messages that failed processing

**Explanation:**
A DLQ is a side queue (or Kafka topic, SQS queue, etc.) where the pipeline routes messages it could not successfully process after retries — e.g., malformed payloads, schema violations, or persistent downstream errors. This isolates "poison pill" messages so they do not block the main pipeline and gives operators a place to inspect, fix, and replay them.

**Common Mistakes:**
- Choosing A (high-priority messages) — priority queues handle ordering, not failure handling.
- Choosing C (auto-delete old messages) — that is retention/TTL policy on a normal queue.
- Choosing D (archive processed messages) — successful messages are typically retained on the source topic or moved to a data lake, not a DLQ.

**Related Material:** `lessons/mod-05-data-pipelines/08-pipeline-monitoring-errors.md`

---

## Question 25
**Q:** True or False: A circuit breaker pattern helps prevent cascading failures by stopping requests to a failing service.

**Answer:** True

**Explanation:**
A circuit breaker tracks the failure rate of calls to a downstream dependency. When failures cross a threshold, the breaker "opens" and short-circuits subsequent calls (fast-failing or serving a fallback) for a cooldown period before transitioning to "half-open" to probe recovery. This prevents resource exhaustion (threads, connections, queues) in the caller and protects the failing service from being overwhelmed by retries — the two mechanisms by which a localized outage normally cascades.

**Common Mistakes:**
- Answering False because the breaker "blocks legitimate traffic" — that short-term cost is *exactly* the trade-off; the alternative is full system collapse.
- Confusing circuit breaker (cross-service protection) with rate limiting (cross-client throttling) or bulkhead (resource isolation). They are complementary resilience patterns, not the same.

**Related Material:** `lessons/mod-05-data-pipelines/08-pipeline-monitoring-errors.md`

---
