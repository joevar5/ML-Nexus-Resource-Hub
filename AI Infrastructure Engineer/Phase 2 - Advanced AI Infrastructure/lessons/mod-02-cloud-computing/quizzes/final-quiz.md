# Module 02: Cloud Computing - Final Quiz

**Time Limit:** 35 minutes
**Passing Score:** 80% (24/30 questions)
**Coverage:** All Lessons (01-08)

---

## Section 1: Cloud Storage for ML (6 questions)

### Q1. What is the main advantage of object storage (S3, GCS) for ML datasets?

a) Fastest access speed
b) Scalability and cost-effectiveness
c) Best for frequent updates
d) Requires no configuration

**Answer:** B

---

### Q2. When should you use block storage (EBS) instead of object storage?

a) For archival data
b) For high-IOPS database workloads
c) For infrequently accessed data
d) Never, always use object storage

**Answer:** B

---

### Q3. What is a data lake?

a) A relational database
b) Centralized repository for structured and unstructured data
c) A caching system
d) A type of VPN

**Answer:** B

---

### Q4. Which caching strategy can improve ML inference performance?

a) Never cache anything
b) Cache predictions for frequent inputs
c) Cache training data only
d) Cache models in RAM only

**Answer:** B

---

### Q5. What is the difference between data lakes and data warehouses?

a) No difference
b) Lakes store raw data, warehouses store processed data
c) Lakes are faster
d) Warehouses are cheaper

**Answer:** B

---

### Q6. Which storage tier is most cost-effective for ML training data accessed monthly?

a) Hot/Standard tier
b) Cool/Infrequent Access tier
c) Archive tier
d) Premium tier

**Answer:** B

---

## Section 2: Cloud Networking for ML (6 questions)

### Q7. What is the purpose of a load balancer in ML systems?

a) Store models
b) Distribute traffic across multiple instances
c) Train models faster
d) Reduce storage costs

**Answer:** B

---

### Q8. Why use a CDN for model serving?

a) To train models faster
b) To reduce latency for global users
c) To increase security only
d) To save on storage costs

**Answer:** B

---

### Q9. What is a service mesh like Istio used for?

a) Training models
b) Managing microservice communication
c) Storing data
d) Monitoring only

**Answer:** B

---

### Q10. What is the purpose of a VPN in hybrid cloud setups?

a) Speed up training
b) Secure connection between on-prem and cloud
c) Reduce costs
d) Store models

**Answer:** B

---

### Q11. Which subnet design is recommended for production ML systems?

a) All resources in one public subnet
b) Public subnet for load balancers, private for compute
c) No subnets needed
d) Random assignment

**Answer:** B

---

### Q12. What is the benefit of using private endpoints for cloud services?

a) Cheaper costs
b) Traffic doesn't traverse public internet
c) Faster training
d) No benefits

**Answer:** B

---

## Section 3: Managed ML Services (6 questions)

### Q13. What is the main advantage of using SageMaker over building custom infrastructure?

a) Always cheaper
b) Reduced operational overhead
c) Better model accuracy
d) Unlimited free tier

**Answer:** B

---

### Q14. What is a Feature Store?

a) App store for ML models
b) Centralized repository for ML features
c) Storage for datasets
d) Model registry

**Answer:** B

---

### Q15. What does a Model Registry provide?

a) Free models
b) Versioning and metadata for models
c) Automatic model training
d) GPU access

**Answer:** B

---

### Q16. Which managed service feature automatically tunes hyperparameters?

a) AutoML
b) Manual tuning
c) Feature engineering
d) Data labeling

**Answer:** A

---

### Q17. What is the trade-off of using managed ML services?

a) No trade-offs
b) Less control vs easier operations
c) Always more expensive
d) Worse performance always

**Answer:** B

---

### Q18. Which cloud provider offers TPU access through their managed ML service?

a) AWS only
b) Google Cloud (Vertex AI)
c) Azure only
d) None of them

**Answer:** B

---

## Section 4: Multi-Cloud and Cost Optimization (12 questions)

### Q19. What is a multi-cloud strategy?

a) Using multiple regions in one cloud
b) Using multiple cloud providers
c) Multiple teams using cloud
d) Multiple accounts in one cloud

**Answer:** B

---

### Q20. What is the main advantage of Reserved Instances?

a) Better performance
b) Significant cost savings (up to 75%)
c) More flexibility
d) Faster deployment

**Answer:** B

---

### Q21. What are Spot Instances best used for?

a) Production databases
b) Fault-tolerant batch processing
c) Real-time inference
d) Critical services

**Answer:** B

---

### Q22. What is the typical discount for Spot Instances?

a) 10-20%
b) 50-90%
c) 5%
d) No discount

**Answer:** B

---

### Q23. Why set up billing alerts in cloud platforms?

a) Required by law
b) Prevent unexpected charges
c) Get better performance
d) Access more services

**Answer:** B

---

### Q24. What is the purpose of cloud cost tagging?

a) Make resources colorful
b) Track spending by project/team/environment
c) Speed up resources
d) No real purpose

**Answer:** B

---

### Q25. Which metric is important for ML cost optimization?

a) Code quality only
b) Cost per training job / inference request
c) Team size
d) Office location

**Answer:** B

---

### Q26. What is the benefit of auto-scaling for ML inference?

a) Better accuracy
b) Match capacity to demand, optimize costs
c) Faster training
d) No benefits

**Answer:** B

---

### Q27. How can you reduce data transfer costs?

a) Never transfer data
b) Keep data and compute in same region
c) Use slowest network
d) Transfer more frequently

**Answer:** B

---

### Q28. What is the typical cost breakdown for ML cloud infrastructure?

a) 90% compute, 10% storage
b) Compute, storage, and networking all significant
c) 100% storage
d) No compute costs

**Answer:** B

---

### Q29. What is a Savings Plan in cloud billing?

a) Free tier extension
b) Commitment to consistent usage for discounts
c) One-time discount
d) Loyalty program

**Answer:** B

---

### Q30. Why use multiple availability zones for production ML systems?

a) Better model accuracy
b) High availability and fault tolerance
c) Cheaper costs
d) Faster training

**Answer:** B

---

## Answer Key

1. B   2. B   3. B   4. B   5. B
6. B   7. B   8. B   9. B   10. B
11. B  12. B  13. B  14. B  15. B
16. A  17. B  18. B  19. B  20. B
21. B  22. B  23. B  24. B  25. B
26. B  27. B  28. B  29. B  30. B

---

## Scoring

- **27-30 correct (90-100%)**: Excellent! Mastered cloud computing
- **24-26 correct (80-89%)**: Good! Ready for next module
- **21-23 correct (70-79%)**: Fair. Review weak areas
- **Below 21 (< 70%)**: Review module materials before continuing

---

## Areas for Review

If you scored poorly in a section, review:

- **Section 1**: Lesson 05 - Cloud Storage for ML
- **Section 2**: Lesson 06 - Cloud Networking for ML
- **Section 3**: Lesson 07 - Managed ML Services
- **Section 4**: Lesson 08 - Multi-Cloud and Cost Optimization

---

## Next Steps

After passing this quiz:

1. Complete the Module 02 Capstone Project
2. Proceed to Module 03: Kubernetes Deep Dive
3. Or explore Module 04: Monitoring and Observability

---

**Congratulations on completing Module 02!**
