# Module 05: Data Pipelines - Hands-On Exercises

## Overview
This directory contains hands-on exercises and labs for Module 05. Complete these exercises to solidify your understanding of data pipelines, Spark, Kafka, and data quality.

---

## Exercise Structure

```
exercises/
├── README.md (this file)
├── lab-01-airflow-basics/
├── lab-02-spark-processing/
├── lab-03-kafka-streaming/
├── lab-04-data-quality/
├── lab-05-end-to-end-pipeline/
└── solutions/
    ├── lab-01/
    ├── lab-02/
    ├── lab-03/
    ├── lab-04/
    └── lab-05/
```

---

## Lab 01: Apache Airflow Basics

**Duration:** 2-3 hours
**Difficulty:** Beginner

### Objectives
- Set up Apache Airflow locally
- Create your first DAG
- Implement task dependencies
- Use XComs for data passing
- Schedule and trigger DAGs

### Setup
```bash
cd lab-01-airflow-basics
./setup.sh
```

### Tasks
1. **Task 1**: Create a simple ETL DAG with 3 tasks
   - Extract data from CSV
   - Transform (filter and aggregate)
   - Load to SQLite database

2. **Task 2**: Add error handling and retries
   - Configure retry strategy
   - Add failure callbacks
   - Implement alerting

3. **Task 3**: Create a branching DAG
   - Use BranchPythonOperator
   - Implement conditional logic
   - Join branches at the end

**Solution:** `solutions/lab-01/`

---

## Lab 02: Spark Data Processing

**Duration:** 3-4 hours
**Difficulty:** Intermediate

### Objectives
- Process large-scale data with PySpark
- Implement feature engineering
- Optimize Spark jobs
- Write output to Parquet

### Dataset
Download sample dataset:
```bash
wget https://example.com/ecommerce-data.csv -O data/ecommerce.csv
# Or use: curl -o data/ecommerce.csv https://example.com/ecommerce-data.csv
```

### Tasks
1. **Task 1**: Basic Spark operations
   - Load 1M+ row dataset
   - Perform filtering and aggregations
   - Compute summary statistics

2. **Task 2**: Feature engineering
   - Create temporal features
   - Compute rolling aggregations
   - Generate user-level features

3. **Task 3**: Performance optimization
   - Optimize partitioning
   - Cache intermediate results
   - Tune Spark configuration

4. **Task 4**: ML pipeline preparation
   - Create feature vectors
   - Split train/test sets
   - Write to Parquet with partitioning

**Starter Code:**
```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# TODO: Create SparkSession

# TODO: Load data

# TODO: Implement transformations

# TODO: Write output
```

**Solution:** `solutions/lab-02/`

---

## Lab 03: Kafka Streaming Pipeline

**Duration:** 3-4 hours
**Difficulty:** Intermediate

### Objectives
- Set up Kafka locally
- Implement producers and consumers
- Build real-time feature engineering
- Handle failures with DLQ

### Setup
```bash
cd lab-03-kafka-streaming
docker-compose up -d
```

### Tasks
1. **Task 1**: Simple producer/consumer
   - Create topic `user-events`
   - Implement event producer
   - Implement basic consumer

2. **Task 2**: Real-time feature engineering
   - Consume events from topic
   - Compute windowed aggregations (1h, 24h)
   - Emit features to output topic

3. **Task 3**: Error handling
   - Implement retry logic
   - Create dead letter queue
   - Add monitoring metrics

4. **Task 4**: Consumer group scaling
   - Run multiple consumer instances
   - Test load balancing
   - Monitor consumer lag

**Architecture:**
```
Producer → [user-events] → Feature Engineer → [features] → ML Model
                                    ↓
                            [dead-letter-queue]
```

**Solution:** `solutions/lab-03/`

---

## Lab 04: Data Quality Framework

**Duration:** 2-3 hours
**Difficulty:** Intermediate

### Objectives
- Implement comprehensive data validation
- Use Great Expectations
- Detect data drift
- Create quality reports

### Tasks
1. **Task 1**: Build custom validator
   - Implement null checks
   - Add range validation
   - Check categorical values
   - Validate data freshness

2. **Task 2**: Great Expectations integration
   - Create expectation suite
   - Define 10+ expectations
   - Generate validation report
   - Set up data docs

3. **Task 3**: Data drift detection
   - Compute baseline statistics
   - Compare new data to baseline
   - Detect statistical drift
   - Generate drift report

4. **Task 4**: Automated quality gates
   - Integrate with CI/CD
   - Fail pipeline on quality issues
   - Send alerts on failures

**Starter Code:**
```python
import great_expectations as gx
import pandas as pd

# TODO: Load data
df = pd.read_csv('data/training_data.csv')

# TODO: Create expectations

# TODO: Validate

# TODO: Generate report
```

**Solution:** `solutions/lab-04/`

---

## Lab 05: End-to-End ML Pipeline

**Duration:** 6-8 hours
**Difficulty:** Advanced

### Objectives
Build a complete production-grade ML data pipeline that demonstrates all concepts from Module 05.

### Requirements
- **Orchestration**: Airflow DAG
- **Data Processing**: Spark for feature engineering
- **Streaming**: Kafka for real-time inference
- **Quality**: Great Expectations validation
- **Monitoring**: Prometheus metrics
- **Versioning**: DVC for data tracking

### Architecture
```
┌─────────────────────────────────────────────────────────┐
│                     Airflow DAG                         │
│                                                          │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐           │
│  │ Extract  │ → │Transform │ → │Validate  │           │
│  │  (API)   │   │ (Spark)  │   │   (GE)   │           │
│  └──────────┘   └──────────┘   └──────────┘           │
│                        ↓                                 │
│                  ┌──────────┐   ┌──────────┐           │
│                  │  Train   │ → │ Register │           │
│                  │  Model   │   │  Model   │           │
│                  └──────────┘   └──────────┘           │
└─────────────────────────────────────────────────────────┘
                          ↓
              ┌───────────────────────┐
              │   Kafka Streaming     │
              │                       │
              │  Events → Features →  │
              │  Model → Predictions  │
              └───────────────────────┘
```

### Phase 1: Batch Training Pipeline (3-4 hours)

**Tasks:**
1. Set up project structure
2. Create Airflow DAG for training
3. Implement Spark feature engineering
4. Add data quality validation
5. Train and register model
6. Version data with DVC

**DAG Structure:**
```python
extract_data >> validate_raw_data >> transform_features >>
validate_features >> train_model >> register_model
```

### Phase 2: Streaming Inference Pipeline (2-3 hours)

**Tasks:**
1. Set up Kafka cluster
2. Implement event producer
3. Build feature engineering consumer
4. Add model serving consumer
5. Emit predictions to output topic

### Phase 3: Monitoring & Observability (1-2 hours)

**Tasks:**
1. Add Prometheus metrics
2. Create Grafana dashboard
3. Implement alerting rules
4. Set up logging with ELK stack

### Success Criteria
- [ ] Pipeline processes 10K+ records
- [ ] Data quality score > 95%
- [ ] Streaming latency < 100ms
- [ ] All tests passing
- [ ] Monitoring dashboard functional
- [ ] Documentation complete

**Solution:** `solutions/lab-05/`

---

## Evaluation Rubric

### Lab 01-04 (25 points each)
- **Functionality** (10 points): Code works correctly
- **Code Quality** (5 points): Clean, readable, documented
- **Best Practices** (5 points): Follows patterns from lessons
- **Testing** (5 points): Includes tests

### Lab 05 (100 points)
- **Architecture** (20 points): Well-designed system
- **Functionality** (30 points): All components work
- **Code Quality** (15 points): Production-ready code
- **Testing** (15 points): Comprehensive test suite
- **Documentation** (10 points): Clear README and comments
- **Monitoring** (10 points): Observability implemented

---

## Getting Help

### Before You Ask
1. Read the lesson materials carefully
2. Check the documentation for the tool you're using
3. Review the starter code and comments
4. Look at similar examples in the lessons

### Where to Ask
- **Discussion Forum**: [Link to forum]
- **Slack Channel**: #data-pipelines-help
- **Office Hours**: Wednesdays 2-4pm PST
- **Stack Overflow**: Tag with `[data-pipeline]`

### Good Questions Include
- What you're trying to accomplish
- What you've tried so far
- Error messages (full stack trace)
- Relevant code snippets
- Your environment (OS, versions)

---

## Submission

### How to Submit
1. Complete the exercises
2. Add your solutions to your GitHub repository
3. Include a README.md with:
   - Setup instructions
   - How to run your code
   - Any assumptions made
   - Challenges encountered
4. Submit the repository URL

### Naming Convention
```
<your-name>-module-05-exercises/
├── lab-01/
├── lab-02/
├── lab-03/
├── lab-04/
├── lab-05/
└── README.md
```

---

## Additional Challenges

Completed all labs? Try these bonus challenges:

### Challenge 1: Multi-Cloud Pipeline
Deploy the same pipeline to AWS, GCP, and Azure. Compare:
- Cost
- Performance
- Operational complexity

### Challenge 2: Real-Time ML
Implement online learning:
- Continuous model updates
- A/B testing infrastructure
- Automated retraining triggers

### Challenge 3: Scale Test
Process 1B+ records:
- Optimize for cost
- Minimize latency
- Handle failures gracefully

### Challenge 4: Advanced Monitoring
Build comprehensive observability:
- Custom Grafana dashboards
- PagerDuty integration
- Automated incident response

---

## Resources

- **Datasets**: See `resources.md` for dataset sources
- **Documentation**: Links to official docs in `resources.md`
- **Video Tutorials**: Recorded walkthroughs available
- **Community**: Join our Slack for peer support

---

## Feedback

Found an issue? Have suggestions?
- Open an issue on GitHub
- Submit a pull request
- Fill out the feedback form: [Link]

---

**Good luck with your exercises! 🚀**

Remember: The goal is to learn, not just to complete. Take your time, experiment, break things, and most importantly - have fun building real data pipelines!
