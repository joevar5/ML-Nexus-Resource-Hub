# Makefile for Project 02: End-to-End MLOps Pipeline
# Provides convenient commands for common development tasks

.PHONY: help setup clean test lint format docker-build docker-run k8s-deploy

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
BLACK := $(PYTHON) -m black
ISORT := $(PYTHON) -m isort
FLAKE8 := $(PYTHON) -m flake8
DOCKER_REGISTRY := localhost:5000
IMAGE_NAME := mlops-model-server
IMAGE_TAG := latest

# ============================================================
# Help
# ============================================================

help:  ## Show this help message
	@echo "MLOps Pipeline - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "TODO: Complete implementation of all make targets"

# ============================================================
# Setup & Installation
# ============================================================

setup:  ## Set up development environment
	@echo "Setting up development environment..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "TODO: Add additional setup steps (DVC init, pre-commit hooks, etc.)"
	# dvc init
	# pre-commit install
	@echo "Setup complete!"

setup-dev:  ## Set up development environment with dev dependencies
	@echo "Setting up development environment with dev tools..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-cov black isort flake8 mypy pylint
	@echo "Development setup complete!"

# ============================================================
# Code Quality
# ============================================================

lint:  ## Run all linters
	@echo "Running linters..."
	@echo "TODO: Implement linting"
	# $(FLAKE8) src/ dags/ tests/
	# $(BLACK) --check src/ dags/ tests/
	# $(ISORT) --check-only src/ dags/ tests/
	# pylint src/ dags/

format:  ## Format code with Black and isort
	@echo "Formatting code..."
	@echo "TODO: Implement code formatting"
	# $(BLACK) src/ dags/ tests/
	# $(ISORT) src/ dags/ tests/

type-check:  ## Run type checking with mypy
	@echo "Running type checking..."
	@echo "TODO: Implement type checking"
	# mypy src/ --ignore-missing-imports

# ============================================================
# Testing
# ============================================================

test:  ## Run all tests
	@echo "Running tests..."
	@echo "TODO: Implement pytest execution"
	# $(PYTEST) tests/ -v

test-data:  ## Run data pipeline tests
	@echo "Running data pipeline tests..."
	@echo "TODO: Implement data pipeline tests"
	# $(PYTEST) tests/test_data.py -v

test-training:  ## Run training pipeline tests
	@echo "Running training pipeline tests..."
	@echo "TODO: Implement training tests"
	# $(PYTEST) tests/test_training.py -v

test-deployment:  ## Run deployment pipeline tests
	@echo "Running deployment tests..."
	@echo "TODO: Implement deployment tests"
	# $(PYTEST) tests/test_deployment.py -v

test-coverage:  ## Run tests with coverage report
	@echo "Running tests with coverage..."
	@echo "TODO: Implement coverage testing"
	# $(PYTEST) tests/ --cov=src --cov-report=html --cov-report=term

# ============================================================
# Docker
# ============================================================

docker-build:  ## Build Docker images
	@echo "Building Docker images..."
	@echo "TODO: Implement Docker build"
	# docker-compose build

docker-up:  ## Start all services with Docker Compose
	@echo "Starting services..."
	docker-compose up -d
	@echo "Services started! Access:"
	@echo "  - Airflow: http://localhost:8080"
	@echo "  - MLflow: http://localhost:5000"
	@echo "  - Prometheus: http://localhost:9090"
	@echo "  - Grafana: http://localhost:3000"

docker-down:  ## Stop all services
	@echo "Stopping services..."
	docker-compose down

docker-logs:  ## View logs from all services
	docker-compose logs -f

docker-clean:  ## Remove all containers, volumes, and images
	@echo "Cleaning Docker resources..."
	docker-compose down -v --rmi all

# ============================================================
# Airflow
# ============================================================

airflow-init:  ## Initialize Airflow database
	@echo "Initializing Airflow..."
	docker-compose run --rm airflow-init

airflow-webserver:  ## Start Airflow webserver
	@echo "Starting Airflow webserver..."
	docker-compose up airflow-webserver

validate-dags:  ## Validate Airflow DAG files
	@echo "Validating DAG files..."
	@echo "TODO: Implement DAG validation"
	# python -c "from airflow.models import DagBag; dagbag = DagBag('dags/'); print(dagbag.import_errors)"

# ============================================================
# MLflow
# ============================================================

mlflow-ui:  ## Start MLflow UI
	@echo "Starting MLflow UI..."
	mlflow ui --host 0.0.0.0 --port 5000

mlflow-server:  ## Start MLflow tracking server
	@echo "Starting MLflow tracking server..."
	@echo "TODO: Configure MLflow server with backend store"
	# mlflow server --backend-store-uri postgresql://mlflow:mlflow@localhost/mlflow \
	#               --default-artifact-root s3://mlflow-artifacts/ \
	#               --host 0.0.0.0 --port 5000

# ============================================================
# Data Pipeline
# ============================================================

data-ingest:  ## Run data ingestion
	@echo "Running data ingestion..."
	@echo "TODO: Implement data ingestion command"
	# $(PYTHON) -m src.data.ingestion

data-validate:  ## Run data validation
	@echo "Running data validation..."
	@echo "TODO: Implement data validation command"
	# $(PYTHON) -m src.data.validation

data-preprocess:  ## Run data preprocessing
	@echo "Running data preprocessing..."
	@echo "TODO: Implement preprocessing command"
	# $(PYTHON) -m src.data.preprocessing

# ============================================================
# Model Training
# ============================================================

train:  ## Train model
	@echo "Training model..."
	@echo "TODO: Implement training command"
	# $(PYTHON) -m src.training.train

evaluate:  ## Evaluate model
	@echo "Evaluating model..."
	@echo "TODO: Implement evaluation command"
	# $(PYTHON) -m src.training.evaluate

# ============================================================
# Kubernetes
# ============================================================

k8s-deploy:  ## Deploy to Kubernetes
	@echo "Deploying to Kubernetes..."
	@echo "TODO: Implement K8s deployment"
	# kubectl apply -f kubernetes/

k8s-delete:  ## Delete Kubernetes resources
	@echo "Deleting Kubernetes resources..."
	# kubectl delete -f kubernetes/

k8s-status:  ## Check deployment status
	@echo "Checking deployment status..."
	# kubectl get pods,svc,deploy -n default

k8s-logs:  ## View pod logs
	@echo "Viewing pod logs..."
	# kubectl logs -f -l app=mlops-model -n default

# ============================================================
# DVC (Data Version Control)
# ============================================================

dvc-init:  ## Initialize DVC
	@echo "Initializing DVC..."
	dvc init

dvc-add-data:  ## Add data to DVC tracking
	@echo "Adding data to DVC..."
	@echo "TODO: Add data files to DVC"
	# dvc add data/raw/data.csv
	# dvc add data/processed/

dvc-push:  ## Push data to remote storage
	@echo "Pushing data to DVC remote..."
	dvc push

dvc-pull:  ## Pull data from remote storage
	@echo "Pulling data from DVC remote..."
	dvc pull

# ============================================================
# Monitoring
# ============================================================

prometheus-up:  ## Start Prometheus
	@echo "Starting Prometheus..."
	docker-compose up -d prometheus

grafana-up:  ## Start Grafana
	@echo "Starting Grafana..."
	docker-compose up -d grafana

# ============================================================
# Cleanup
# ============================================================

clean:  ## Clean temporary files and caches
	@echo "Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".coverage" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/
	@echo "Cleanup complete!"

clean-all: clean docker-clean  ## Clean everything including Docker resources
	@echo "Deep cleanup complete!"

# ============================================================
# Documentation
# ============================================================

docs:  ## Generate documentation
	@echo "Generating documentation..."
	@echo "TODO: Implement documentation generation"
	# sphinx-build -b html docs/ docs/_build/

# ============================================================
# Utilities
# ============================================================

requirements-update:  ## Update requirements.txt from environment
	@echo "Updating requirements.txt..."
	$(PIP) freeze > requirements.txt

check-env:  ## Check if environment is properly set up
	@echo "Checking environment..."
	@$(PYTHON) --version
	@$(PIP) --version
	@echo "TODO: Add more environment checks"
