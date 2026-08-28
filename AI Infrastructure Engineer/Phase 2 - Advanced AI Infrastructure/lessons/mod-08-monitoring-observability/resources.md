# Module 08: Monitoring & Observability - Resources

## Official Documentation

### Prometheus
- **Main Documentation:** https://prometheus.io/docs/
- **PromQL Guide:** https://prometheus.io/docs/prometheus/latest/querying/basics/
- **Best Practices:** https://prometheus.io/docs/practices/naming/
- **Alerting Rules:** https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/
- **Recording Rules:** https://prometheus.io/docs/prometheus/latest/configuration/recording_rules/

### Grafana
- **Documentation:** https://grafana.com/docs/
- **Dashboard Best Practices:** https://grafana.com/docs/grafana/latest/dashboards/build-dashboards/best-practices/
- **Provisioning:** https://grafana.com/docs/grafana/latest/administration/provisioning/
- **Pre-built Dashboards:** https://grafana.com/grafana/dashboards/
- **Community Forum:** https://community.grafana.com/

### Elasticsearch & ELK Stack
- **Elasticsearch:** https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html
- **Logstash:** https://www.elastic.co/guide/en/logstash/current/index.html
- **Kibana:** https://www.elastic.co/guide/en/kibana/current/index.html
- **Filebeat:** https://www.elastic.co/guide/en/beats/filebeat/current/index.html
- **ELK Getting Started:** https://www.elastic.co/guide/en/elastic-stack-get-started/current/index.html

### Grafana Loki
- **Documentation:** https://grafana.com/docs/loki/latest/
- **LogQL Language:** https://grafana.com/docs/loki/latest/logql/
- **Best Practices:** https://grafana.com/docs/loki/latest/best-practices/
- **Promtail:** https://grafana.com/docs/loki/latest/clients/promtail/

### Distributed Tracing
- **OpenTelemetry:** https://opentelemetry.io/docs/
- **Jaeger:** https://www.jaegertracing.io/docs/
- **Grafana Tempo:** https://grafana.com/docs/tempo/latest/
- **OpenTelemetry Python:** https://opentelemetry.io/docs/instrumentation/python/

### Alertmanager
- **Documentation:** https://prometheus.io/docs/alerting/latest/
- **Configuration:** https://prometheus.io/docs/alerting/latest/configuration/
- **Notification Templates:** https://prometheus.io/docs/alerting/latest/notifications/

---

## Books

### Observability & Monitoring

1. **"Observability Engineering" by Charity Majors, Liz Fong-Jones, George Miranda**
   - Publisher: O'Reilly (2022)
   - Focus: Building observable systems, culture, and practices
   - Level: Intermediate to Advanced
   - [O'Reilly Link](https://www.oreilly.com/library/view/observability-engineering/9781492076438/)

2. **"Distributed Systems Observability" by Cindy Sridharan**
   - Publisher: O'Reilly (2018)
   - Focus: Monitoring, logging, and tracing in distributed systems
   - Level: Intermediate
   - [O'Reilly Link](https://www.oreilly.com/library/view/distributed-systems-observability/9781492033431/)

3. **"Practical Monitoring" by Mike Julian**
   - Publisher: O'Reilly (2017)
   - Focus: Effective monitoring strategies and practices
   - Level: Beginner to Intermediate
   - [O'Reilly Link](https://www.oreilly.com/library/view/practical-monitoring/9781491957349/)

4. **"Prometheus: Up & Running" by Brian Brazil**
   - Publisher: O'Reilly (2018)
   - Focus: Comprehensive Prometheus guide
   - Level: Beginner to Intermediate
   - [O'Reilly Link](https://www.oreilly.com/library/view/prometheus-up/9781492034131/)

### Site Reliability Engineering

5. **"Site Reliability Engineering" (The SRE Book) by Google**
   - Publisher: O'Reilly (2016)
   - Focus: Google's approach to production systems
   - Level: Intermediate to Advanced
   - Free online: https://sre.google/sre-book/table-of-contents/
   - Key chapters: Monitoring Distributed Systems, Practical Alerting

6. **"The Site Reliability Workbook" by Google**
   - Publisher: O'Reilly (2018)
   - Focus: Practical implementation of SRE principles
   - Free online: https://sre.google/workbook/table-of-contents/
   - Key chapters: SLO Engineering, Alerting on SLOs

---

## Online Courses

### Free Courses

1. **Prometheus & Grafana Tutorials (TechWorld with Nana)**
   - Platform: YouTube
   - Link: https://www.youtube.com/c/TechWorldwithNana
   - Duration: Multiple videos (2-4 hours total)
   - Level: Beginner

2. **Observability with Grafana (Grafana Labs)**
   - Platform: Grafana Learning
   - Link: https://grafana.com/tutorials/
   - Duration: Self-paced
   - Level: Beginner to Intermediate

3. **ELK Stack Tutorial (DigitalOcean)**
   - Platform: DigitalOcean Community
   - Link: https://www.digitalocean.com/community/tutorial_series/centralized-logging-with-elk-stack
   - Duration: Self-paced
   - Level: Beginner

### Paid Courses

4. **"Monitoring and Alerting with Prometheus" (Linux Academy/ACG)**
   - Platform: A Cloud Guru
   - Duration: 3-4 hours
   - Level: Intermediate
   - Includes: Hands-on labs

5. **"Complete Observability with Grafana Stack" (Udemy)**
   - Platform: Udemy
   - Duration: 6-8 hours
   - Level: Intermediate
   - Covers: Prometheus, Grafana, Loki, Tempo

6. **"Site Reliability Engineering: Measuring and Managing Reliability" (Coursera)**
   - Platform: Coursera (Google Cloud)
   - Duration: 4 weeks
   - Level: Intermediate

---

## Interactive Labs & Playgrounds

1. **Prometheus Playground**
   - Link: https://play.prometheus.io/
   - Description: Online Prometheus instance with sample data
   - Use: Practice PromQL queries

2. **Grafana Play**
   - Link: https://play.grafana.org/
   - Description: Live Grafana instance with sample dashboards
   - Use: Explore dashboards and features

3. **Katacoda Scenarios (archived, now O'Reilly)**
   - Platform: O'Reilly Learning
   - Topics: Prometheus, Grafana, ELK, Kubernetes monitoring
   - Level: Beginner to Intermediate

4. **Killercoda**
   - Link: https://killercoda.com/
   - Topics: Various observability scenarios
   - Level: Beginner to Intermediate

---

## Tools & Libraries

### Python Libraries

```bash
# Prometheus client
pip install prometheus-client

# OpenTelemetry
pip install opentelemetry-api opentelemetry-sdk
pip install opentelemetry-instrumentation-fastapi
pip install opentelemetry-exporter-prometheus
pip install opentelemetry-exporter-jaeger

# Structured logging
pip install python-json-logger
pip install structlog

# ML monitoring
pip install evidently  # Data drift detection
pip install mlflow     # Experiment tracking
pip install shap       # Model explainability
```

### Exporters

- **Node Exporter:** https://github.com/prometheus/node_exporter
- **NVIDIA GPU Exporter (DCGM):** https://github.com/NVIDIA/dcgm-exporter
- **Kubernetes Metrics:** https://github.com/kubernetes/kube-state-metrics
- **Blackbox Exporter:** https://github.com/prometheus/blackbox_exporter
- **PostgreSQL Exporter:** https://github.com/prometheus-community/postgres_exporter
- **MongoDB Exporter:** https://github.com/percona/mongodb_exporter

### Visualization & Analysis

- **Grafana Dashboards:** https://grafana.com/grafana/dashboards/
  - GPU Monitoring: Dashboard #12239
  - Node Exporter: Dashboard #1860
  - Kubernetes: Dashboard #315

---

## Blogs & Articles

### Observability Best Practices

1. **"Metrics, Tracing, and Logging" by Peter Bourgon**
   - Link: https://peter.bourgon.org/blog/2017/02/21/metrics-tracing-and-logging.html
   - Topic: Distinguishing the three pillars

2. **"Monitoring in the Time of Cloud Native" (CNCF Blog)**
   - Link: https://www.cncf.io/blog/
   - Topic: Modern monitoring approaches

3. **"How to Monitor the SRE Golden Signals" (Google Cloud)**
   - Link: https://cloud.google.com/blog/products/devops-sre
   - Topic: Four Golden Signals implementation

### ML-Specific Monitoring

4. **"ML Model Monitoring in Production" (Eugene Yan)**
   - Link: https://eugeneyan.com/writing/ml-monitoring/
   - Topic: Comprehensive ML monitoring guide

5. **"Data Drift in Machine Learning" (Evidently AI Blog)**
   - Link: https://www.evidentlyai.com/blog
   - Topic: Data drift detection and monitoring

6. **"MLOps Monitoring Best Practices" (Neptune.ai)**
   - Link: https://neptune.ai/blog/ml-model-monitoring-best-tools
   - Topic: Tools and practices for ML monitoring

### Company Engineering Blogs

7. **Uber Engineering Blog**
   - Link: https://eng.uber.com/
   - Topics: Jaeger, M3, observability at scale

8. **Netflix Tech Blog**
   - Link: https://netflixtechblog.com/
   - Topics: Atlas, observability patterns

9. **Spotify Engineering**
   - Link: https://engineering.atspotify.com/
   - Topics: ML monitoring, data quality

---

## GitHub Repositories

### Example Projects

1. **Awesome Prometheus**
   - Link: https://github.com/roaldnefs/awesome-prometheus
   - Description: Curated list of Prometheus resources

2. **Grafana Dashboards Collection**
   - Link: https://github.com/rfmoz/grafana-dashboards
   - Description: Community dashboard collection

3. **OpenTelemetry Demo**
   - Link: https://github.com/open-telemetry/opentelemetry-demo
   - Description: Microservices demo with full observability

4. **ML Monitoring Examples**
   - Link: https://github.com/evidentlyai/evidently
   - Description: ML monitoring and drift detection

### Reference Implementations

5. **kube-prometheus-stack**
   - Link: https://github.com/prometheus-community/helm-charts/tree/main/charts/kube-prometheus-stack
   - Description: Complete monitoring stack for Kubernetes

6. **Prometheus Operator**
   - Link: https://github.com/prometheus-operator/prometheus-operator
   - Description: Kubernetes operator for Prometheus

---

## Conferences & Talks

### Annual Conferences

1. **KubeCon + CloudNativeCon**
   - Topics: Observability in cloud-native environments
   - Videos: https://www.youtube.com/c/cloudnativefdn

2. **ObservabilityCON (Grafana Labs)**
   - Topics: Latest Grafana ecosystem developments
   - Videos: https://grafana.com/about/events/observabilitycon/

3. **PromCon (Prometheus Conference)**
   - Topics: Prometheus ecosystem and best practices
   - Videos: https://www.youtube.com/c/PrometheusIo

### Notable Talks

4. **"Observability: A 3-Year Retrospective" - Charity Majors**
   - Conference: QCon
   - Link: Search on InfoQ

5. **"Distributed Tracing for Microservices" - Ben Sigelman**
   - Conference: Various
   - Link: Search on YouTube

---

## Communities

### Forums & Discussion

1. **Prometheus Community**
   - Mailing list: https://groups.google.com/forum/#!forum/prometheus-users
   - Slack: https://prometheus.io/community/

2. **Grafana Community**
   - Forum: https://community.grafana.com/
   - Slack: https://grafana.slack.com/

3. **CNCF Slack**
   - Observability channels
   - Join: https://communityinviter.com/apps/cloud-native/cncf

4. **r/devops (Reddit)**
   - Link: https://www.reddit.com/r/devops/
   - Topics: Monitoring, observability discussions

---

## Certifications

### Relevant Certifications

1. **Certified Kubernetes Administrator (CKA)**
   - Includes: Monitoring Kubernetes clusters
   - Provider: CNCF
   - Link: https://training.linuxfoundation.org/certification/certified-kubernetes-administrator-cka/

2. **Prometheus Certified Associate (PCA)**
   - Focus: Prometheus fundamentals
   - Provider: Linux Foundation
   - Link: https://training.linuxfoundation.org/certification/prometheus-certified-associate/

3. **Site Reliability Engineering (SRE) Certification**
   - Focus: SRE practices including monitoring
   - Provider: DevOps Institute
   - Link: https://www.devopsinstitute.com/certifications/sre-foundation/

---

## Cheat Sheets

1. **PromQL Cheat Sheet**
   - Link: https://promlabs.com/promql-cheat-sheet/

2. **LogQL Cheat Sheet**
   - Link: https://grafana.com/docs/loki/latest/logql/

3. **Elasticsearch Query DSL**
   - Link: https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl.html

---

## Newsletters

1. **SRE Weekly**
   - Link: https://sreweekly.com/
   - Frequency: Weekly
   - Topics: SRE, monitoring, incidents

2. **DevOps'ish**
   - Link: https://devopsish.com/
   - Frequency: Weekly
   - Topics: DevOps, cloud-native, observability

3. **Last Week in AWS**
   - Link: https://www.lastweekinaws.com/
   - Frequency: Weekly
   - Topics: AWS, including monitoring services

---

## Practice Projects

### Beginner Projects

1. **Set up Prometheus + Grafana locally**
   - Monitor a sample application
   - Create custom dashboards
   - Configure basic alerts

2. **Implement structured logging**
   - Add JSON logging to a Python app
   - Send logs to Loki or ELK
   - Create log queries

### Intermediate Projects

3. **Build a complete observability stack**
   - Metrics: Prometheus
   - Logs: Loki or ELK
   - Traces: Jaeger or Tempo
   - Dashboards: Grafana

4. **ML Model Monitoring**
   - Instrument an ML inference service
   - Track accuracy, latency, throughput
   - Implement drift detection
   - Create ML-specific dashboards

### Advanced Projects

5. **Multi-cluster Observability**
   - Federated Prometheus setup
   - Centralized logging across clusters
   - Distributed tracing across services

6. **Custom Exporter Development**
   - Build a Prometheus exporter for a custom system
   - Implement proper metric naming
   - Add configuration options

---

## Additional Tools to Explore

### Observability Platforms

- **Datadog:** https://www.datadoghq.com/
- **New Relic:** https://newrelic.com/
- **Dynatrace:** https://www.dynatrace.com/
- **Honeycomb:** https://www.honeycomb.io/
- **Lightstep:** https://lightstep.com/

### Open Source Alternatives

- **VictoriaMetrics:** Prometheus-compatible, better performance
- **Thanos:** Long-term Prometheus storage
- **Cortex:** Horizontally scalable Prometheus
- **M3:** Uber's metrics platform
- **SigNoz:** Open-source observability platform

---

## Keep Learning!

Observability is a rapidly evolving field. Stay updated by:
- Following key maintainers on Twitter/X
- Reading release notes for tools you use
- Experimenting with new features
- Sharing your learnings with the community
- Contributing to open-source projects

**Happy monitoring!** 📊🔍🚀
