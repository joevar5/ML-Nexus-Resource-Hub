# Project 03: LLM Deployment Platform - Makefile
# Quick commands for development, testing, and deployment

.PHONY: help setup install test lint format run deploy clean benchmark

# Default target
help:
	@echo "LLM Deployment Platform - Available Commands:"
	@echo "  make setup       - Initial setup (create dirs, download model)"
	@echo "  make install     - Install Python dependencies"
	@echo "  make test        - Run all tests"
	@echo "  make lint        - Run linters (flake8, mypy)"
	@echo "  make format      - Format code (black, isort)"
	@echo "  make run         - Run application locally"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run  - Run Docker container"
	@echo "  make deploy      - Deploy to Kubernetes"
	@echo "  make benchmark   - Run performance benchmarks"
	@echo "  make monitor     - Start monitoring stack"
	@echo "  make clean       - Clean temporary files"

# Setup
setup:
	@echo "Setting up LLM Deployment Platform..."
	mkdir -p data/sample_documents models logs
	chmod +x scripts/*.sh
	@echo "Run './scripts/setup.sh' to download model and setup environment"

# Install dependencies
install:
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "Dependencies installed!"

# Run tests
test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "Tests complete! Coverage report in htmlcov/index.html"

# Linting
lint:
	flake8 src/ tests/ --max-line-length=120
	mypy src/ --ignore-missing-imports
	@echo "Linting complete!"

# Code formatting
format:
	black src/ tests/ --line-length=120
	isort src/ tests/ --profile black
	@echo "Code formatted!"

# Run application locally
run:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Docker build
docker-build:
	docker build -t llm-deployment:latest -f Dockerfile .
	@echo "Docker image built: llm-deployment:latest"

# Docker run
docker-run:
	docker-compose up -d
	@echo "Services running! API: http://localhost:8000/docs"

# Docker stop
docker-stop:
	docker-compose down
	@echo "Services stopped"

# Deploy to Kubernetes
deploy:
	kubectl apply -f kubernetes/
	@echo "Deployed to Kubernetes!"

# Deploy monitoring
monitor:
	kubectl apply -f monitoring/
	@echo "Monitoring deployed!"

# Run benchmarks
benchmark:
	@echo "Running performance benchmarks..."
	chmod +x scripts/benchmark.sh
	./scripts/benchmark.sh

# Clean
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage
	@echo "Cleaned temporary files!"

# Development helpers
dev-setup: setup install
	@echo "Development environment ready!"

# Full local test
local-test: format lint test
	@echo "All local tests passed!"

# Quick start
quickstart: install docker-build docker-run
	@echo "LLM Deployment Platform started!"
	@echo "API Documentation: http://localhost:8000/docs"
	@echo "Grafana: http://localhost:3000 (admin/admin)"
