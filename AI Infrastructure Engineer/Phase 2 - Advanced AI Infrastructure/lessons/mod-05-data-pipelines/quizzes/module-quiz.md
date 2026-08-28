# Module 05: Data Pipelines - Quiz

## Instructions
- This quiz covers all lessons in Module 05 (Data Pipelines)
- 25 questions total covering Apache Airflow, DVC, Spark, Kafka, Data Quality, and Monitoring
- Mix of multiple choice, true/false, and short answer questions
- Passing score: 80% (20/25 correct)

---

## Section 1: Apache Airflow Fundamentals (Questions 1-6)

### Question 1
What is the primary purpose of Apache Airflow in ML infrastructure?

A) To train machine learning models
B) To orchestrate and schedule complex data workflows
C) To serve model predictions
D) To store training data

**Answer:** B

---

### Question 2
In Airflow, what does DAG stand for?

A) Data Aggregation Graph
B) Directed Acyclic Graph
C) Dynamic Algorithm Generator
D) Distributed Application Gateway

**Answer:** B

---

### Question 3
True or False: In Airflow, tasks within a DAG can have circular dependencies.

**Answer:** False (DAG must be acyclic - no cycles allowed)

---

### Question 4
Which Airflow component is responsible for executing tasks?

A) Scheduler
B) Web Server
C) Executor
D) Metadata Database

**Answer:** C

---

### Question 5
What is the purpose of the `depends_on_past` parameter in Airflow?

A) To prevent tasks from running if the previous DAG run failed
B) To share data between tasks
C) To schedule tasks in the past
D) To check task dependencies

**Answer:** A

---

### Question 6 (Short Answer)
Explain the difference between `CeleryExecutor` and `LocalExecutor` in Airflow.

**Sample Answer:**
- **LocalExecutor**: Runs tasks in parallel on a single machine using multiprocessing. Good for development and small-scale deployments.
- **CeleryExecutor**: Distributes tasks across multiple worker machines using Celery. Suitable for production environments requiring horizontal scalability.

---

## Section 2: Data Versioning with DVC (Questions 7-10)

### Question 7
What problem does DVC (Data Version Control) solve?

A) Managing large datasets with Git
B) Tracking model performance
C) Deploying models to production
D) Monitoring data quality

**Answer:** A

---

### Question 8
True or False: DVC stores the actual data files in Git repositories.

**Answer:** False (DVC stores metadata/pointers in Git, actual data in remote storage)

---

### Question 9
Which command is used to track a new dataset with DVC?

A) `dvc add <file>`
B) `dvc track <file>`
C) `dvc commit <file>`
D) `dvc push <file>`

**Answer:** A

---

### Question 10
What is the purpose of a DVC pipeline?

A) To clean data
B) To define reproducible data processing workflows
C) To compress large files
D) To encrypt sensitive data

**Answer:** B

---

## Section 3: Apache Spark (Questions 11-15)

### Question 11
What is the main advantage of Apache Spark over traditional MapReduce?

A) Better security
B) In-memory processing for 100x speedup
C) Smaller storage requirements
D) Easier installation

**Answer:** B

---

### Question 12
In Spark, what is a DataFrame?

A) A distributed collection of data organized into named columns
B) A single-machine pandas DataFrame
C) A SQL database table
D) A JSON file format

**Answer:** A

---

### Question 13
True or False: In Spark, transformations are executed immediately when called.

**Answer:** False (Transformations are lazy and only executed when an action is called)

---

### Question 14
Which of the following is a Spark action (not a transformation)?

A) `filter()`
B) `map()`
C) `groupBy()`
D) `count()`

**Answer:** D

---

### Question 15 (Short Answer)
Explain what "shuffle" means in Spark and why it's expensive.

**Sample Answer:**
A shuffle is the process of redistributing data across partitions, typically triggered by operations like `groupBy()`, `join()`, or `reduceByKey()`. It's expensive because:
- Requires disk I/O to write intermediate results
- Involves network transfer of data between executors
- Can cause memory pressure and spilling to disk
- Adds latency to job execution

---

## Section 4: Apache Kafka (Questions 16-19)

### Question 16
What is Apache Kafka primarily used for?

A) Batch data processing
B) Real-time event streaming
C) Model training
D) Data warehousing

**Answer:** B

---

### Question 17
In Kafka, what is a topic?

A) A database table
B) A category or feed name to which messages are published
C) A consumer instance
D) A configuration file

**Answer:** B

---

### Question 18
True or False: In Kafka, consumers in the same consumer group will receive duplicate messages.

**Answer:** False (Messages are distributed among consumers in a group - load balancing)

---

### Question 19
What is the purpose of partitions in Kafka?

A) To encrypt messages
B) To enable parallel processing and scalability
C) To compress data
D) To validate message format

**Answer:** B

---

## Section 5: Data Quality & Validation (Questions 20-22)

### Question 20
Which of the following is NOT a dimension of data quality?

A) Completeness
B) Accuracy
C) Velocity
D) Consistency

**Answer:** C (Velocity is from the 3 Vs of Big Data, not a quality dimension)

---

### Question 21
What is data drift in the context of ML?

A) Data being corrupted during transmission
B) Statistical properties of data changing over time
C) Data being deleted accidentally
D) Data being stored in the wrong location

**Answer:** B

---

### Question 22 (Short Answer)
Name three common data quality checks you should implement in an ML training pipeline.

**Sample Answer:**
1. **Null/Missing Value Checks**: Ensure critical columns don't have excessive nulls
2. **Range Validation**: Check numerical values are within expected bounds (e.g., age between 0-120)
3. **Schema Validation**: Verify expected columns exist with correct data types
4. **Uniqueness Checks**: Ensure ID columns don't have duplicates
5. **Distribution Checks**: Detect if data distribution has shifted significantly

(Any 3 are acceptable)

---

## Section 6: Monitoring & Error Handling (Questions 23-25)

### Question 23
What are the three pillars of observability?

A) Metrics, Logs, Dashboards
B) Metrics, Logs, Traces
C) Alerts, Logs, Reports
D) Metrics, Alerts, Traces

**Answer:** B

---

### Question 24
What is a Dead Letter Queue (DLQ)?

A) A queue for high-priority messages
B) A storage location for messages that failed processing
C) A queue that automatically deletes old messages
D) A queue for archiving processed messages

**Answer:** B

---

### Question 25
True or False: A circuit breaker pattern helps prevent cascading failures by stopping requests to a failing service.

**Answer:** True

---

## Scoring Guide

**Points Distribution:**
- Section 1 (Airflow): 6 points
- Section 2 (DVC): 4 points
- Section 3 (Spark): 5 points
- Section 4 (Kafka): 4 points
- Section 5 (Data Quality): 3 points
- Section 6 (Monitoring): 3 points

**Total: 25 points**

**Grading:**
- 23-25: Excellent (92-100%)
- 20-22: Very Good (80-88%)
- 17-19: Good (68-76%)
- 14-16: Satisfactory (56-64%)
- Below 14: Needs Improvement

---

## Additional Practice

For more practice, try these activities:

1. **Hands-on Lab**: Complete the exercises in the `exercises/` directory
2. **Build a Pipeline**: Create an end-to-end pipeline using Airflow + Spark + Kafka
3. **Data Quality Project**: Implement comprehensive data validation using Great Expectations
4. **Monitoring Setup**: Set up Prometheus + Grafana for pipeline monitoring
5. **Real-world Challenge**: Process a large dataset (> 1GB) with Spark and track with DVC

---

## Answer Key Location

Full answer keys live alongside each module's quiz in this repo (`module-quiz-answers.md` where present). Worked examples for the practical questions are in the paired [`ai-infra-engineer-solutions`](https://github.com/ai-infra-curriculum/ai-infra-engineer-solutions/tree/main/modules/mod-105-data-pipelines) repo.

**Note:** For short answer questions, answers may vary. The sample answers provided are examples of acceptable responses. Focus on demonstrating understanding of core concepts rather than exact wording.
