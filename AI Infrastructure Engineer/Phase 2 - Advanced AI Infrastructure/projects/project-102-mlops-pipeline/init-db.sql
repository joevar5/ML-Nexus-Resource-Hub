-- PostgreSQL Initialization Script
-- Creates databases and users for MLflow and Airflow

-- This script runs automatically when PostgreSQL container starts for the first time
-- Reference: docker-compose.yml postgres service

-- ============================================================
-- MLflow Database Setup
-- ============================================================

-- Create MLflow database
CREATE DATABASE mlflow;

-- Create MLflow user
-- TODO: Replace with secure password from environment variable
CREATE USER mlflow WITH PASSWORD 'mlflow';

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow;

-- Connect to mlflow database and set up schema
\c mlflow

-- Grant schema privileges
GRANT ALL ON SCHEMA public TO mlflow;

-- TODO: Create MLflow tables (usually done by MLflow migrations)
-- MLflow will create tables automatically on first run

-- ============================================================
-- Airflow Database Setup
-- ============================================================

-- Switch back to postgres database
\c postgres

-- Create Airflow database (if not exists from docker-compose)
-- The main database is already created via POSTGRES_DB env var
-- So we just need to set up proper permissions

-- Grant privileges to airflow user
GRANT ALL PRIVILEGES ON DATABASE airflow TO airflow;

-- Connect to airflow database
\c airflow

-- Grant schema privileges
GRANT ALL ON SCHEMA public TO airflow;

-- ============================================================
-- Create additional databases if needed
-- ============================================================

-- TODO: Create additional databases for other services
-- Example: Feature store, metadata store, etc.

/*
-- Feature Store Database (Example)
\c postgres
CREATE DATABASE feature_store;
CREATE USER featurestore WITH PASSWORD 'featurestore';
GRANT ALL PRIVILEGES ON DATABASE feature_store TO featurestore;

\c feature_store
GRANT ALL ON SCHEMA public TO featurestore;
*/

-- ============================================================
-- Create Extensions
-- ============================================================

-- Install useful PostgreSQL extensions

\c mlflow
-- Extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Extension for full-text search (if needed)
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

\c airflow
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- ============================================================
-- Performance Tuning (Optional)
-- ============================================================

-- TODO: Add performance tuning settings
-- These should match your workload characteristics

/*
ALTER DATABASE mlflow SET work_mem = '32MB';
ALTER DATABASE mlflow SET maintenance_work_mem = '128MB';
ALTER DATABASE mlflow SET effective_cache_size = '1GB';

ALTER DATABASE airflow SET work_mem = '32MB';
ALTER DATABASE airflow SET maintenance_work_mem = '128MB';
*/

-- ============================================================
-- Create Indexes (Optional)
-- ============================================================

-- TODO: Add custom indexes for performance
-- MLflow and Airflow will create their own indexes

-- ============================================================
-- Monitoring Setup
-- ============================================================

-- TODO: Set up pg_stat_statements for query monitoring
/*
\c mlflow
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

\c airflow
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
*/

-- ============================================================
-- Backup Configuration
-- ============================================================

-- TODO: Document backup strategy
/*
Recommended backup strategy:
1. Automated daily backups using pg_dump
2. Point-in-time recovery (PITR) with WAL archiving
3. Periodic full backups
4. Test restore procedures regularly

Example backup command:
pg_dump -U mlflow -F c -f mlflow_backup.dump mlflow
pg_dump -U airflow -F c -f airflow_backup.dump airflow
*/

-- ============================================================
-- Security Hardening
-- ============================================================

-- TODO: Implement security best practices
/*
1. Use strong passwords (read from secrets)
2. Restrict network access (pg_hba.conf)
3. Enable SSL/TLS connections
4. Regular security updates
5. Principle of least privilege for users
6. Audit logging
*/

-- ============================================================
-- Maintenance Tasks
-- ============================================================

-- TODO: Set up regular maintenance
/*
Recommended maintenance tasks:
1. VACUUM ANALYZE (automatically handled by autovacuum)
2. REINDEX (if needed)
3. Monitor table bloat
4. Monitor connection limits
5. Review slow queries
*/

-- Completion message
DO $$
BEGIN
    RAISE NOTICE 'Database initialization completed successfully!';
    RAISE NOTICE 'MLflow database: mlflow (user: mlflow)';
    RAISE NOTICE 'Airflow database: airflow (user: airflow)';
END
$$;
