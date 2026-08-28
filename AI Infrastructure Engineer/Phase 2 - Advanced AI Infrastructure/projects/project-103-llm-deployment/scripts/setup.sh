#!/bin/bash
set -e

echo "Setting up LLM Deployment Platform..."

# TODO: Check prerequisites
echo "Checking prerequisites..."
command -v python3 >/dev/null 2>&1 || { echo "Python 3 required but not installed."; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "Docker required but not installed."; exit 1; }

# TODO: Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# TODO: Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# TODO: Download model (if needed)
echo "Downloading model..."
# python -c "from transformers import AutoModel; AutoModel.from_pretrained('meta-llama/Llama-2-7b-chat-hf')"

# TODO: Setup vector database
echo "Setting up vector database..."
# docker-compose up -d vector-db

# TODO: Create configuration
echo "Creating configuration..."
cp .env.example .env
echo "Please edit .env with your API keys"

# TODO: Run tests
echo "Running tests..."
pytest tests/ -v

echo "Setup complete!"
echo "Run 'source venv/bin/activate' to activate the environment"
echo "Run 'python src/api/main.py' to start the server"
