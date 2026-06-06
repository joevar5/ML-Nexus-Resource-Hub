# Exercise 01: Python Environment Setup & Automation

## Overview
This exercise covers setting up a reproducible development environment for a Sentiment Classification project. You will structure a project, configure virtual environments, pin dependencies, manage configurations securely with environment variables, and automate the setup.

* **Time**: 30-45 mins
* **Level**: Beginner

---

## Part 1: Project Structure & Git Integration

A clean layout keeps training pipelines, configurations, and test suites organized.

Create the following files and directories:

```text
sentiment-classifier/
├── .gitignore
├── requirements.txt
├── requirements-dev.txt
├── setup.sh
├── .env.example
├── src/
│   ├── __init__.py
│   ├── train.py
│   └── utils/
│       ├── __init__.py
│       └── metrics.py
└── tests/
    ├── __init__.py
    └── test_metrics.py
```

### Git Ignored Patterns (`.gitignore`)
Configure your `.gitignore` to avoid checking in virtual environments, secret credentials, cached files, or large trained model artifacts:

```gitignore
.venv/
__pycache__/
*.pyc
.env
*.pt
*.h5
data/
*.log
```

---

## Part 2: Dependency & Environment Management

Pinning exact versions avoids "works on my machine" compatibility errors.

### 1. Production Requirements (`requirements.txt`)
```txt
numpy==1.24.3
pandas==2.1.0
torch==2.1.0
transformers==4.35.0
python-dotenv==1.0.0
```

### 2. Development Requirements (`requirements-dev.txt`)
```txt
-r requirements.txt
pytest==7.4.3
black==23.11.0
mypy==1.7.0
```

---

## Part 3: Environment Configuration

Store API keys, model names, and batch sizes in environment variables rather than hardcoding them in scripts.

### 1. Configuration Template (`.env.example`)
```ini
MODEL_NAME=bert-base-uncased
BATCH_SIZE=32
DEVICE=cpu
LOG_LEVEL=INFO
```

### 2. Loading Variables (`test_env.py`)
```python
import os
from dotenv import load_dotenv

# Load key-value pairs from .env into environment variables
load_dotenv()

model = os.getenv("MODEL_NAME", "default-model")
batch_size = int(os.getenv("BATCH_SIZE", "16"))

print(f"Model configured: {model}")
print(f"Batch Size configured: {batch_size}")
```

---

## Part 4: Automation (`setup.sh`)

Automate the installation so other developers can get started in one click.

Create `setup.sh`:
```bash
#!/bin/bash
set -e  # Exit instantly if a command fails

echo "🚀 Setting up Sentiment Classifier development environment..."

# 1. Create virtual environment
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

# 2. Activate environment
source .venv/bin/activate

# 3. Upgrade package installer and install dependencies
pip install --upgrade pip
pip install -r requirements-dev.txt

# 4. Copy configuration template if .env doesn't exist
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "📄 Created .env configuration file."
fi

echo "✅ Setup complete. Run 'source .venv/bin/activate' to begin."
```

**Your Turn:**
1. Make `setup.sh` executable by running: `chmod +x setup.sh`
2. Run it: `./setup.sh`

---

## Quick Checklist
1. Why should you never commit `.env` files to Git?
2. What is the difference between `requirements.txt` and `requirements-dev.txt`?
3. How do you activate the virtual environment on Windows vs. macOS?
