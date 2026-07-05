# Project 01: Simple Model API Deployment

---

## The Scenario: Bridging Research and Production

Imagine you have just joined **NexusAI**, a fast-growing startup, as their first Junior AI Infrastructure Engineer. 

For the past three months, the Data Science team has been training a state-of-the-art image classification model. They are thrilled because it achieves 95% accuracy in notebook environments. However, right now, that model is sitting idle as a raw file in a shared storage bucket. It cannot help a single real user.

This morning, the Product Manager rushes to your desk: 
> "Our mobile app developers are ready to integrate the new image-tagging feature, but they have no way to talk to the model. We need a fast, reliable REST API where they can send an image and get predictions back. And we need it containerized and running in the cloud by next week!"

This is your mission. You are the bridge between machine learning research and the real world. Your code will take a static model file and turn it into a living, breathing, production-grade cloud service.

---

## Your Scope of Work

To succeed in this mission, you must understand your goals from both a functional perspective and a system-level architectural perspective.

### 1. Functional Scope (What the Solution Must Do)
First, you need to align on what functions the API must perform. You will implement:
*   **Model Loading & Preprocessing:** Safely load the model weights at startup and preprocess incoming images (resize, color channel adjustments, normalization).
*   **Prediction Endpoint:** A secure `POST /predict` route that accepts multipart image uploads and returns the top 5 classifications.
*   **Operational Endpoints:** A `/health` check to monitor service status and an `/info` endpoint for model versioning.

For a detailed list of specifications and acceptance criteria, check out the [requirements.md](requirements.md) file.

### 2. Architectural Scope (How the Solution Fits Together)
Next, you need to design how this service will run reliably in production:
*   **Stateless Component Design:** Separating configuration, model loading, and server routing to make the application easy to test and maintain.
*   **Containerization:** Packaging the application with Docker to guarantee that "it works on my machine" matches "it works in the cloud."
*   **Deployment:** Provisioning a cloud VM, configuring firewall rules (security groups), and setting up logging to monitor performance under load.

For the system diagrams, design trade-offs, and scaling considerations, consult the [architecture.md](architecture.md) guide.

---

## Project Structure

This workspace is structured to help you build the service layer by layer:

```bash
project-01-simple-model-api/
├── README.md                      # This file
├── requirements.md                # Detailed requirements specification
├── architecture.md                # System architecture and design
├── .env.example                   # Environment variable template
├── src/
│   ├── README.md                  # Code structure guide
│   ├── app.py                     # Main Flask/FastAPI application (STUB)
│   ├── model_loader.py            # Model loading logic (STUB)
│   └── config.py                  # Configuration management (STUB)
├── tests/
│   ├── test_app.py                # API endpoint tests (STUB)
│   └── test_model.py              # Model functionality tests (STUB)
└── docker/
    ├── Dockerfile                 # Container definition (STUB)
    └── docker-compose.yml         # Local development setup
```

> [!TIP]  
> You do not need to build everything from scratch. The files in `src/` are pre-configured stubs with type hints and detailed `TODO` comments to guide your implementation.

---

## Quick Start

### 1. Set Up Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install flask fastapi uvicorn torch torchvision pillow python-dotenv pytest requests
cp .env.example .env
```

### 2. Run and Test Locally
```bash
# Start the local server
python src/app.py

# Verify the health endpoint in a new terminal
curl http://localhost:5000/health

# Run the test suite
pytest tests/
```
