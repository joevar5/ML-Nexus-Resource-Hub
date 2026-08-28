# Lesson 04: Azure for ML Infrastructure

**Duration:** 6 hours
**Difficulty:** Intermediate
**Prerequisites:** Lesson 01-03 (Cloud Architecture, AWS, GCP)

## Learning Objectives

By the end of this lesson, you will be able to:

1. **Set up and secure** an Azure account with proper identity management
2. **Deploy and manage** Virtual Machines for ML workloads
3. **Use Azure Blob Storage** for datasets and models
4. **Deploy models** on Azure Kubernetes Service (AKS)
5. **Leverage Azure Machine Learning** for managed ML workflows
6. **Integrate Azure OpenAI** services into applications
7. **Configure networking** for secure ML infrastructure
8. **Optimize costs** using Azure pricing models

---

## Table of Contents

1. [Introduction to Azure for ML](#introduction-to-azure-for-ml)
2. [Azure Account Setup and Identity Management](#azure-account-setup-and-identity-management)
3. [Virtual Machines for ML](#virtual-machines-for-ml)
4. [Azure Blob Storage](#azure-blob-storage)
5. [Azure Kubernetes Service (AKS)](#azure-kubernetes-service-aks)
6. [Azure Machine Learning](#azure-machine-learning)
7. [Azure OpenAI Service](#azure-openai-service)
8. [Azure Networking](#azure-networking)
9. [Cost Optimization](#cost-optimization)
10. [Hands-on Exercise](#hands-on-exercise)

---

## Introduction to Azure for ML

Microsoft Azure offers a comprehensive set of services for ML infrastructure:

- **Enterprise focus**: Strong integration with Microsoft ecosystem
- **Azure OpenAI**: Exclusive access to GPT-4, GPT-3.5, DALL-E
- **Hybrid cloud**: Seamless on-premises integration with Azure Arc
- **Azure ML**: End-to-end managed ML platform
- **Strong compliance**: GDPR, HIPAA, SOC certifications
- **Global presence**: 60+ regions worldwide

### Azure ML Ecosystem

```
┌──────────────────────────────────────────────────────────────────┐
│                      Azure ML Ecosystem                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Data Services        Compute              ML Services          │
│  ─────────────        ───────              ───────────          │
│  Blob Storage    →    Virtual Machines →   Azure ML Studio     │
│  Data Lake       →    AKS              →   Cognitive Services  │
│  SQL Database    →    Azure Batch      →   Azure OpenAI       │
│                                                                  │
│  Infrastructure       Networking           Monitoring           │
│  ──────────────       ──────────           ──────────           │
│  Resource Groups      VNet                 Azure Monitor       │
│  Subscriptions        Load Balancer        Application Insights│
│  Management Groups    CDN                  Log Analytics       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### When to Choose Azure for ML

**Choose Azure when:**
- You're in a Microsoft-centric organization
- You need Azure OpenAI (GPT-4, ChatGPT)
- You require strong hybrid cloud capabilities
- You need enterprise compliance and governance
- You want tight integration with Office 365, Teams, Power BI

**Consider alternatives when:**
- You're already heavily invested in AWS/GCP
- You need more specialized ML hardware (TPUs)
- You prefer open-source-first ecosystems
- Cost is primary concern (Azure can be more expensive)

---

## Azure Account Setup and Identity Management

### Creating an Azure Account

1. **Sign up for Azure**:
   - Visit: https://azure.microsoft.com/free/
   - Get **$200 credit** valid for 30 days
   - 12 months of popular services free
   - 55+ always-free services

2. **Free tier benefits**:
   ```
   Compute:
   - 750 hours B1S VM (Linux/Windows) per month
   - 1M Azure Functions requests
   - 20 compute hours Azure Container Instances

   Storage:
   - 5 GB Blob Storage (Hot tier)
   - 5 GB File Storage
   - 2 million reads, 2 million writes

   Databases:
   - 250 GB SQL Database
   - 5 GB Cosmos DB

   Machine Learning:
   - Azure ML compute hours (varied)
   - 5,000 transactions Cognitive Services
   ```

3. **Install Azure CLI**:
   ```bash
   # macOS
   brew install azure-cli

   # Linux
   curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

   # Windows
   # Download from: https://aka.ms/installazurecliwindows

   # Verify installation
   az --version

   # Login
   az login

   # Set subscription (if you have multiple)
   az account list --output table
   az account set --subscription "My Subscription"
   ```

### Azure Active Directory (Azure AD)

Azure uses **Azure Active Directory** for identity and access management.

#### Key Concepts

```
┌────────────────────────────────────────────────────────┐
│              Azure AD Hierarchy                        │
├────────────────────────────────────────────────────────┤
│                                                        │
│  Tenant (Organization)                                 │
│    ├── Subscription 1                                  │
│    │     ├── Resource Group A                          │
│    │     │     ├── Virtual Machine                     │
│    │     │     └── Storage Account                     │
│    │     └── Resource Group B                          │
│    │           └── AKS Cluster                         │
│    └── Subscription 2                                  │
│          └── Resource Group C                          │
│                └── Azure ML Workspace                  │
│                                                        │
└────────────────────────────────────────────────────────┘
```

#### Creating Resource Groups

Resource Groups are logical containers for Azure resources:

```bash
# Create resource group
az group create \
  --name ml-infrastructure-rg \
  --location eastus

# List resource groups
az group list --output table

# Delete resource group (and all resources)
az group delete --name ml-infrastructure-rg --yes
```

#### Role-Based Access Control (RBAC)

```bash
# List available roles
az role definition list --output table

# Assign role to user
az role assignment create \
  --assignee user@example.com \
  --role "Contributor" \
  --scope /subscriptions/{subscription-id}/resourceGroups/ml-infrastructure-rg

# Common ML roles:
# - Contributor: Full access except granting access to others
# - Reader: View all resources but can't make changes
# - Owner: Full access including managing access
# - AcrPull: Pull images from Azure Container Registry
# - Storage Blob Data Contributor: Read, write, delete blob containers and data
```

#### Creating Service Principals

Service principals are used for application authentication:

```bash
# Create service principal
az ad sp create-for-rbac \
  --name ml-training-sp \
  --role Contributor \
  --scopes /subscriptions/{subscription-id}/resourceGroups/ml-infrastructure-rg

# Output:
# {
#   "appId": "xxxx-xxxx-xxxx-xxxx",
#   "displayName": "ml-training-sp",
#   "password": "xxxx-xxxx-xxxx-xxxx",
#   "tenant": "xxxx-xxxx-xxxx-xxxx"
# }

# Use in applications
export AZURE_CLIENT_ID="<appId>"
export AZURE_CLIENT_SECRET="<password>"
export AZURE_TENANT_ID="<tenant>"
export AZURE_SUBSCRIPTION_ID="<subscription-id>"
```

---

## Virtual Machines for ML

Azure Virtual Machines provide scalable compute for ML workloads.

### VM Series for ML

#### General Purpose (D-series)

Best for: Development, small models, CPU inference

```
┌─────────────────┬─────────┬────────────┬──────────────┐
│ VM Size         │ vCPUs   │ Memory     │ Cost/hour    │
├─────────────────┼─────────┼────────────┼──────────────┤
│ Standard_D4s_v3 │ 4       │ 16 GB      │ $0.192       │
│ Standard_D8s_v3 │ 8       │ 32 GB      │ $0.384       │
│ Standard_D16s_v3│ 16      │ 64 GB      │ $0.768       │
└─────────────────┴─────────┴────────────┴──────────────┘
```

#### GPU VMs (NC/ND/NV-series)

Best for: Training, GPU inference

```
┌─────────────────┬──────────────────┬────────────┬─────────┬──────────────┐
│ VM Size         │ GPU              │ GPU Memory │ vCPUs   │ Cost/hour    │
├─────────────────┼──────────────────┼────────────┼─────────┼──────────────┤
│ Standard_NC4    │ 1x Tesla K80     │ 12 GB      │ 4       │ $0.90        │
│ Standard_NC6s_v3│ 1x Tesla V100    │ 16 GB      │ 6       │ $3.06        │
│ Standard_ND6s   │ 1x Tesla P40     │ 24 GB      │ 6       │ $2.07        │
│ Standard_NC6s_v3│ 1x V100          │ 16 GB      │ 6       │ $3.06        │
│ Standard_ND40rs │ 8x V100          │ 128 GB     │ 40      │ $24.48       │
│ Standard_NC24ads│ 1x A100          │ 80 GB      │ 24      │ $3.67        │
└─────────────────┴──────────────────┴────────────┴─────────┴──────────────┘
```

### Creating a GPU VM

```bash
# List available VM sizes with GPUs
az vm list-sizes --location eastus --output table | grep NC

# Create GPU VM with Data Science VM image
az vm create \
  --resource-group ml-infrastructure-rg \
  --name ml-training-vm \
  --image microsoft-dsvm:ubuntu-1804:1804-gen2:latest \
  --size Standard_NC6s_v3 \
  --admin-username azureuser \
  --generate-ssh-keys \
  --public-ip-sku Standard

# SSH into VM
az vm show \
  --resource-group ml-infrastructure-rg \
  --name ml-training-vm \
  --show-details \
  --query publicIps \
  --output tsv

ssh azureuser@<public-ip>

# Verify GPU
nvidia-smi
```

### Data Science Virtual Machine (DSVM)

Azure DSVM comes pre-installed with ML frameworks:

- **Frameworks**: PyTorch, TensorFlow, scikit-learn
- **Tools**: Jupyter, VS Code, PyCharm
- **Data tools**: Azure CLI, AzCopy, Azure Storage Explorer
- **Development**: Git, Docker, Kubernetes tools

```bash
# Create DSVM (Ubuntu)
az vm create \
  --resource-group ml-infrastructure-rg \
  --name dsvm-gpu \
  --image microsoft-dsvm:ubuntu-2004:2004-gen2:latest \
  --size Standard_NC6s_v3 \
  --admin-username azureuser \
  --generate-ssh-keys

# Create DSVM (Windows)
az vm create \
  --resource-group ml-infrastructure-rg \
  --name dsvm-windows \
  --image microsoft-dsvm:dsvm-win-2019:server-2019:latest \
  --size Standard_NC6s_v3 \
  --admin-username azureuser \
  --admin-password <secure-password>
```

### Spot VMs (Low-Priority VMs)

Save up to **90%** with Azure Spot VMs:

```bash
# Create Spot VM
az vm create \
  --resource-group ml-infrastructure-rg \
  --name ml-training-spot \
  --image microsoft-dsvm:ubuntu-1804:1804-gen2:latest \
  --size Standard_NC6s_v3 \
  --priority Spot \
  --max-price 0.5 \
  --eviction-policy Deallocate \
  --admin-username azureuser \
  --generate-ssh-keys

# Cost comparison:
# Regular NC6s_v3: $3.06/hour
# Spot NC6s_v3: ~$0.30-0.60/hour (80-90% savings)
```

### VM Startup Script

```bash
# Create startup script
cat > startup.sh << 'EOF'
#!/bin/bash

# Update system
apt-get update
apt-get upgrade -y

# Install Python packages
pip install torch torchvision wandb mlflow

# Download dataset
mkdir -p /data
azcopy copy \
  "https://mystorageaccount.blob.core.windows.net/datasets/*" \
  "/data/" \
  --recursive

# Clone training repo
git clone https://github.com/myorg/ml-training.git /home/azureuser/training

# Start training
cd /home/azureuser/training
python train.py --data-path /data --epochs 100
EOF

# Create VM with startup script
az vm create \
  --resource-group ml-infrastructure-rg \
  --name ml-training-auto \
  --image microsoft-dsvm:ubuntu-1804:1804-gen2:latest \
  --size Standard_NC6s_v3 \
  --custom-data startup.sh \
  --admin-username azureuser \
  --generate-ssh-keys
```

### Checkpointing for Spot VMs

```python
import os
import signal
import sys
import torch
from azure.storage.blob import BlobServiceClient

# Azure Blob Storage setup
connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client("checkpoints")

def save_checkpoint_to_blob(state, filename):
    """Save checkpoint to Azure Blob Storage"""
    local_path = f"/tmp/{filename}"
    torch.save(state, local_path)

    blob_client = container_client.get_blob_client(filename)
    with open(local_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)

    print(f"Checkpoint saved to blob: {filename}")
    os.remove(local_path)

def load_checkpoint_from_blob(filename):
    """Load checkpoint from Azure Blob Storage"""
    local_path = f"/tmp/{filename}"
    blob_client = container_client.get_blob_client(filename)

    if blob_client.exists():
        with open(local_path, "wb") as f:
            blob_data = blob_client.download_blob()
            blob_data.readinto(f)

        checkpoint = torch.load(local_path)
        os.remove(local_path)
        return checkpoint

    return None

def signal_handler(signum, frame):
    """Handle eviction signal"""
    print("Spot VM eviction detected, saving checkpoint...")
    save_checkpoint_to_blob({
        'epoch': current_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': current_loss,
    }, 'checkpoint_evicted.pth')
    sys.exit(0)

# Register signal handler for Azure Spot VM eviction
signal.signal(signal.SIGTERM, signal_handler)

# Training loop with Azure Blob checkpointing
def train_with_azure_checkpointing(model, train_loader, epochs=10):
    # Try to resume from checkpoint
    checkpoint = load_checkpoint_from_blob("latest.pth")
    start_epoch = 0

    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Save checkpoint every 100 batches
            if batch_idx % 100 == 0:
                save_checkpoint_to_blob({
                    'epoch': epoch,
                    'batch': batch_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, "latest.pth")

        print(f"Epoch {epoch} completed")
```

---

## Azure Blob Storage

Azure Blob Storage is object storage for unstructured data.

### Storage Account Types

```
┌──────────────────────┬────────────────┬────────────────────────┐
│ Performance Tier     │ Redundancy     │ Use Case               │
├──────────────────────┼────────────────┼────────────────────────┤
│ Standard (HDD)       │ LRS            │ Backup, archival       │
│ Standard (HDD)       │ GRS            │ Geo-redundant backup   │
│ Premium (SSD)        │ LRS            │ High-throughput ML     │
└──────────────────────┴────────────────┴────────────────────────┘

Redundancy Options:
- LRS (Locally Redundant): 3 copies in one datacenter
- ZRS (Zone Redundant): 3 copies across availability zones
- GRS (Geo-Redundant): 6 copies (3 local + 3 remote)
- GZRS (Geo-Zone Redundant): 6 copies across zones and regions
```

### Access Tiers and Pricing

```
┌────────────────┬───────────────┬──────────────────┬────────────────┐
│ Tier           │ Storage Cost  │ Access Cost      │ Use Case       │
├────────────────┼───────────────┼──────────────────┼────────────────┤
│ Hot            │ $0.0184/GB    │ Low              │ Active data    │
│ Cool           │ $0.01/GB      │ Medium           │ <30 days       │
│ Archive        │ $0.00099/GB   │ High             │ >180 days      │
└────────────────┴───────────────┴──────────────────┴────────────────┘
```

### Creating Storage Account

```bash
# Create storage account
az storage account create \
  --name mlstorageacct001 \
  --resource-group ml-infrastructure-rg \
  --location eastus \
  --sku Standard_LRS \
  --kind StorageV2

# Get connection string
az storage account show-connection-string \
  --name mlstorageacct001 \
  --resource-group ml-infrastructure-rg \
  --output tsv

# Create container
az storage container create \
  --name datasets \
  --account-name mlstorageacct001

az storage container create \
  --name models \
  --account-name mlstorageacct001
```

### Using AzCopy for Large Files

```bash
# Install AzCopy
wget https://aka.ms/downloadazcopy-v10-linux
tar -xvf downloadazcopy-v10-linux
sudo cp azcopy_linux_amd64_*/azcopy /usr/bin/

# Get SAS token
az storage container generate-sas \
  --account-name mlstorageacct001 \
  --name datasets \
  --permissions acdlrw \
  --expiry 2024-12-31 \
  --output tsv

# Upload file
azcopy copy "model.pth" \
  "https://mlstorageacct001.blob.core.windows.net/models/model.pth?<SAS-token>"

# Upload directory (parallel)
azcopy copy "./datasets" \
  "https://mlstorageacct001.blob.core.windows.net/datasets?<SAS-token>" \
  --recursive

# Download file
azcopy copy \
  "https://mlstorageacct001.blob.core.windows.net/models/model.pth?<SAS-token>" \
  "./model.pth"

# Sync directories (like rsync)
azcopy sync "./local-dir" \
  "https://mlstorageacct001.blob.core.windows.net/datasets?<SAS-token>" \
  --recursive
```

### Python SDK (azure-storage-blob)

```python
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import os

# Initialize client
connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

# Create container
container_name = "models"
container_client = blob_service_client.create_container(container_name)

# Upload file
def upload_file_to_blob(local_file, blob_name):
    """Upload file to Azure Blob Storage"""
    blob_client = blob_service_client.get_blob_client(
        container=container_name,
        blob=blob_name
    )

    with open(local_file, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)

    print(f"Uploaded {local_file} to {blob_name}")

# Upload model
upload_file_to_blob("model.pth", "resnet50-v1.pth")

# Download file
def download_file_from_blob(blob_name, local_file):
    """Download file from Azure Blob Storage"""
    blob_client = blob_service_client.get_blob_client(
        container=container_name,
        blob=blob_name
    )

    with open(local_file, "wb") as f:
        blob_data = blob_client.download_blob()
        blob_data.readinto(f)

    print(f"Downloaded {blob_name} to {local_file}")

# Download model
download_file_from_blob("resnet50-v1.pth", "./model.pth")

# List blobs
def list_blobs(container_name, prefix=None):
    """List blobs in container"""
    container_client = blob_service_client.get_container_client(container_name)
    blobs = container_client.list_blobs(name_starts_with=prefix)

    for blob in blobs:
        print(f"Name: {blob.name}, Size: {blob.size} bytes")

# List all models
list_blobs("models", prefix="resnet")

# Generate SAS URL (temporary access)
from azure.storage.blob import generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta

def generate_sas_url(container_name, blob_name, expiry_hours=24):
    """Generate SAS URL for temporary access"""
    account_name = "mlstorageacct001"
    account_key = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")

    sas_token = generate_blob_sas(
        account_name=account_name,
        container_name=container_name,
        blob_name=blob_name,
        account_key=account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(hours=expiry_hours)
    )

    url = f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_name}?{sas_token}"
    return url

# Get temporary URL for model
url = generate_sas_url("models", "resnet50-v1.pth", expiry_hours=48)
print(f"Temporary URL (48 hours): {url}")

# Copy blob within Azure
def copy_blob(source_container, source_blob, dest_container, dest_blob):
    """Copy blob within Azure Storage"""
    source_blob_client = blob_service_client.get_blob_client(source_container, source_blob)
    dest_blob_client = blob_service_client.get_blob_client(dest_container, dest_blob)

    dest_blob_client.start_copy_from_url(source_blob_client.url)
    print(f"Copied {source_blob} to {dest_blob}")

# Promote model from staging to production
copy_blob("staging", "resnet50-v2.pth", "production", "resnet50-latest.pth")
```

### Lifecycle Management

```bash
# Create lifecycle policy (JSON)
cat > lifecycle-policy.json << EOF
{
  "rules": [
    {
      "enabled": true,
      "name": "move-to-cool",
      "type": "Lifecycle",
      "definition": {
        "actions": {
          "baseBlob": {
            "tierToCool": {
              "daysAfterModificationGreaterThan": 30
            },
            "tierToArchive": {
              "daysAfterModificationGreaterThan": 90
            },
            "delete": {
              "daysAfterModificationGreaterThan": 365
            }
          }
        },
        "filters": {
          "blobTypes": ["blockBlob"],
          "prefixMatch": ["datasets/"]
        }
      }
    }
  ]
}
EOF

# Apply lifecycle policy
az storage account management-policy create \
  --account-name mlstorageacct001 \
  --resource-group ml-infrastructure-rg \
  --policy @lifecycle-policy.json
```

---

## Azure Kubernetes Service (AKS)

AKS is Azure's managed Kubernetes service for container orchestration.

### Creating an AKS Cluster

```bash
# Create AKS cluster
az aks create \
  --resource-group ml-infrastructure-rg \
  --name ml-aks-cluster \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3 \
  --enable-addons monitoring \
  --generate-ssh-keys

# Get credentials
az aks get-credentials \
  --resource-group ml-infrastructure-rg \
  --name ml-aks-cluster

# Verify
kubectl get nodes
```

### Creating AKS Cluster with GPU Nodes

```bash
# Create AKS cluster with GPU node pool
az aks create \
  --resource-group ml-infrastructure-rg \
  --name ml-gpu-aks-cluster \
  --node-count 1 \
  --node-vm-size Standard_D4s_v3 \
  --generate-ssh-keys

# Add GPU node pool
az aks nodepool add \
  --resource-group ml-infrastructure-rg \
  --cluster-name ml-gpu-aks-cluster \
  --name gpupool \
  --node-count 2 \
  --node-vm-size Standard_NC6s_v3 \
  --node-taints sku=gpu:NoSchedule

# Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml

# Verify GPUs
kubectl get nodes -o json | jq '.items[].status.capacity'
```

### Deploying ML Model on AKS

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-model
  template:
    metadata:
      labels:
        app: ml-model
    spec:
      containers:
      - name: model-server
        image: myacr.azurecr.io/ml-model:v1.0
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: "https://mlstorageacct001.blob.core.windows.net/models/model.pth"
        - name: AZURE_STORAGE_CONNECTION_STRING
          valueFrom:
            secretKeyRef:
              name: azure-storage-secret
              key: connection-string
        resources:
          requests:
            cpu: 1000m
            memory: 2Gi
          limits:
            cpu: 2000m
            memory: 4Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: ml-model-service
spec:
  type: LoadBalancer
  selector:
    app: ml-model
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
```

Deploy:
```bash
# Create secret for Azure Storage
kubectl create secret generic azure-storage-secret \
  --from-literal=connection-string="<connection-string>"

# Deploy
kubectl apply -f deployment.yaml

# Check status
kubectl get deployments
kubectl get pods
kubectl get services
```

### GPU Deployment on AKS

```yaml
# gpu-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-gpu-deployment
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ml-gpu-model
  template:
    metadata:
      labels:
        app: ml-gpu-model
    spec:
      nodeSelector:
        kubernetes.io/hostname: gpu-node  # Schedule on GPU nodes
      tolerations:
      - key: sku
        operator: Equal
        value: gpu
        effect: NoSchedule
      containers:
      - name: gpu-model-server
        image: myacr.azurecr.io/ml-model-gpu:v1.0
        resources:
          requests:
            nvidia.com/gpu: 1
          limits:
            nvidia.com/gpu: 1
```

### AKS Autoscaling

```bash
# Enable cluster autoscaler
az aks update \
  --resource-group ml-infrastructure-rg \
  --name ml-aks-cluster \
  --enable-cluster-autoscaler \
  --min-count 1 \
  --max-count 10

# Horizontal Pod Autoscaler (HPA)
kubectl autoscale deployment ml-model-deployment \
  --cpu-percent=70 \
  --min=2 \
  --max=20
```

---

## Azure Machine Learning

Azure Machine Learning is a comprehensive managed platform for ML workflows.

### Creating Azure ML Workspace

```bash
# Create ML workspace
az ml workspace create \
  --name ml-workspace \
  --resource-group ml-infrastructure-rg \
  --location eastus

# Get workspace details
az ml workspace show \
  --name ml-workspace \
  --resource-group ml-infrastructure-rg
```

### Training with Azure ML

```python
from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core.compute import ComputeTarget, AmlCompute

# Connect to workspace
ws = Workspace.from_config()  # Reads from config.json
# Or:
ws = Workspace.get(
    name='ml-workspace',
    subscription_id='<subscription-id>',
    resource_group='ml-infrastructure-rg'
)

# Create compute cluster
compute_name = "gpu-cluster"
compute_config = AmlCompute.provisioning_configuration(
    vm_size='Standard_NC6s_v3',
    max_nodes=4,
    idle_seconds_before_scaledown=300
)

compute_target = ComputeTarget.create(ws, compute_name, compute_config)
compute_target.wait_for_completion(show_output=True)

# Create environment
env = Environment.from_conda_specification(
    name='pytorch-env',
    file_path='environment.yml'
)

# Or use curated environment
env = Environment.get(ws, name='AzureML-PyTorch-1.13-CUDA11.6')

# Create training script config
config = ScriptRunConfig(
    source_directory='./src',
    script='train.py',
    arguments=[
        '--data-path', ws.datasets['imagenet'].as_mount(),
        '--epochs', 100,
        '--batch-size', 64,
        '--learning-rate', 0.001
    ],
    compute_target=compute_target,
    environment=env
)

# Submit experiment
experiment = Experiment(ws, 'resnet-training')
run = experiment.submit(config)

# Monitor run
run.wait_for_completion(show_output=True)

# Download model
run.download_file(
    name='outputs/model.pth',
    output_file_path='./model.pth'
)
```

### Deploying Model with Azure ML

```python
from azureml.core.model import Model, InferenceConfig
from azureml.core.webservice import AciWebservice, AksWebservice

# Register model
model = Model.register(
    workspace=ws,
    model_name='resnet50',
    model_path='./model.pth',
    description='ResNet-50 trained on ImageNet',
    tags={'framework': 'pytorch', 'task': 'classification'}
)

# Create inference config
inference_config = InferenceConfig(
    entry_script='score.py',
    environment=env
)

# Deploy to Azure Container Instances (ACI) - for testing
aci_config = AciWebservice.deploy_configuration(
    cpu_cores=2,
    memory_gb=4,
    auth_enabled=True
)

service = Model.deploy(
    workspace=ws,
    name='resnet50-aci',
    models=[model],
    inference_config=inference_config,
    deployment_config=aci_config
)

service.wait_for_deployment(show_output=True)
print(f"Scoring URI: {service.scoring_uri}")

# Deploy to AKS - for production
aks_target = ComputeTarget(ws, 'ml-aks-cluster')

aks_config = AksWebservice.deploy_configuration(
    autoscale_enabled=True,
    autoscale_min_replicas=2,
    autoscale_max_replicas=10,
    autoscale_target_utilization=70,
    cpu_cores=2,
    memory_gb=4,
    enable_app_insights=True
)

service = Model.deploy(
    workspace=ws,
    name='resnet50-production',
    models=[model],
    inference_config=inference_config,
    deployment_config=aks_config,
    deployment_target=aks_target
)

service.wait_for_deployment(show_output=True)

# Test endpoint
import requests
import json

headers = {'Content-Type': 'application/json'}
headers['Authorization'] = f'Bearer {service.get_keys()[0]}'

data = {'data': [[1, 2, 3, 4, 5]]}
response = requests.post(service.scoring_uri, json=data, headers=headers)
print(response.json())
```

### Azure ML Pipelines

```python
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep

# Define pipeline data
processed_data = PipelineData('processed', datastore=ws.get_default_datastore())
trained_model = PipelineData('model', datastore=ws.get_default_datastore())

# Data preprocessing step
preprocess_step = PythonScriptStep(
    name='preprocess-data',
    script_name='preprocess.py',
    arguments=['--output', processed_data],
    outputs=[processed_data],
    compute_target=compute_target,
    source_directory='./src'
)

# Training step
train_step = PythonScriptStep(
    name='train-model',
    script_name='train.py',
    arguments=['--input', processed_data, '--output', trained_model],
    inputs=[processed_data],
    outputs=[trained_model],
    compute_target=compute_target,
    source_directory='./src'
)

# Evaluation step
evaluate_step = PythonScriptStep(
    name='evaluate-model',
    script_name='evaluate.py',
    arguments=['--model', trained_model],
    inputs=[trained_model],
    compute_target=compute_target,
    source_directory='./src'
)

# Create pipeline
pipeline = Pipeline(
    workspace=ws,
    steps=[preprocess_step, train_step, evaluate_step]
)

# Submit pipeline
experiment = Experiment(ws, 'ml-pipeline')
pipeline_run = experiment.submit(pipeline)
pipeline_run.wait_for_completion(show_output=True)

# Publish pipeline for reuse
published_pipeline = pipeline.publish(
    name='training-pipeline-v1',
    description='Complete training pipeline'
)
```

---

## Azure OpenAI Service

Azure OpenAI provides exclusive access to OpenAI's models through Azure.

### Available Models

- **GPT-4**: Most capable model, best for complex tasks
- **GPT-3.5-Turbo**: Fast, cost-effective for most tasks
- **GPT-3.5-Turbo-16k**: Extended context window (16k tokens)
- **DALL-E 3**: Image generation
- **Whisper**: Speech-to-text
- **Embeddings**: Text embeddings for similarity search

### Creating Azure OpenAI Resource

```bash
# Create Azure OpenAI resource
az cognitiveservices account create \
  --name my-openai-resource \
  --resource-group ml-infrastructure-rg \
  --kind OpenAI \
  --sku S0 \
  --location eastus

# Get API key
az cognitiveservices account keys list \
  --name my-openai-resource \
  --resource-group ml-infrastructure-rg
```

### Using Azure OpenAI with Python

```python
import openai
import os

# Set up Azure OpenAI
openai.api_type = "azure"
openai.api_base = "https://my-openai-resource.openai.azure.com/"
openai.api_version = "2023-05-15"
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")

# Chat completion (GPT-4)
response = openai.ChatCompletion.create(
    engine="gpt-4",  # Deployment name in Azure
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Explain machine learning in simple terms"}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response['choices'][0]['message']['content'])

# Embeddings
response = openai.Embedding.create(
    engine="text-embedding-ada-002",
    input="Machine learning is a subset of artificial intelligence"
)

embedding = response['data'][0]['embedding']
print(f"Embedding dimension: {len(embedding)}")

# Image generation (DALL-E)
response = openai.Image.create(
    prompt="A futuristic AI data center with glowing servers",
    n=1,
    size="1024x1024"
)

image_url = response['data'][0]['url']
print(f"Generated image: {image_url}")
```

### Integrating Azure OpenAI with ML Pipeline

```python
import openai
from azureml.core import Workspace, Dataset

# Connect to Azure ML workspace
ws = Workspace.from_config()

# Load dataset
dataset = Dataset.get_by_name(ws, 'customer-feedback')
df = dataset.to_pandas_dataframe()

# Analyze sentiments with GPT-4
def analyze_sentiment(text):
    response = openai.ChatCompletion.create(
        engine="gpt-4",
        messages=[
            {"role": "system", "content": "Analyze sentiment: positive, negative, or neutral"},
            {"role": "user", "content": text}
        ],
        temperature=0,
        max_tokens=10
    )
    return response['choices'][0]['message']['content']

# Apply to dataset
df['sentiment'] = df['feedback'].apply(analyze_sentiment)

# Register updated dataset
updated_dataset = Dataset.Tabular.register_pandas_dataframe(
    df,
    target=(ws.get_default_datastore(), 'feedback-with-sentiment'),
    name='customer-feedback-analyzed'
)
```

### Azure OpenAI Pricing

```
┌──────────────────────┬────────────────────────────────────────┐
│ Model                │ Price                                  │
├──────────────────────┼────────────────────────────────────────┤
│ GPT-4 (8K context)   │ $0.03/1K prompt + $0.06/1K completion  │
│ GPT-4 (32K context)  │ $0.06/1K prompt + $0.12/1K completion  │
│ GPT-3.5-Turbo        │ $0.0015/1K prompt + $0.002/1K compl    │
│ GPT-3.5-Turbo-16k    │ $0.003/1K prompt + $0.004/1K compl     │
│ Embeddings           │ $0.0001/1K tokens                      │
│ DALL-E 3             │ $0.04-0.12 per image                   │
│ Whisper              │ $0.006/minute                          │
└──────────────────────┴────────────────────────────────────────┘
```

---

## Azure Networking

### Virtual Network (VNet) Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                   Azure Virtual Network                      │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Subnet 1: Training (10.0.1.0/24)                           │
│  ├── GPU VM 1                                                │
│  ├── GPU VM 2                                                │
│  └── DSVM                                                    │
│                                                              │
│  Subnet 2: Inference (10.0.2.0/24)                          │
│  ├── AKS Cluster                                             │
│  │   ├── Pod 1                                               │
│  │   ├── Pod 2                                               │
│  │   └── Pod 3                                               │
│  └── Load Balancer                                           │
│                                                              │
│  Subnet 3: Data (10.0.3.0/24)                               │
│  └── Private Endpoint → Blob Storage                        │
│                                                              │
└──────────────────────────────────────────────────────────────┘
         ▲                               ▲
         │                               │
    Azure Bastion                 Application Gateway
    (Secure SSH)                  (Public HTTPS)
```

### Creating VNet

```bash
# Create VNet
az network vnet create \
  --resource-group ml-infrastructure-rg \
  --name ml-vnet \
  --address-prefix 10.0.0.0/16 \
  --subnet-name training-subnet \
  --subnet-prefix 10.0.1.0/24

# Add additional subnets
az network vnet subnet create \
  --resource-group ml-infrastructure-rg \
  --vnet-name ml-vnet \
  --name inference-subnet \
  --address-prefix 10.0.2.0/24

az network vnet subnet create \
  --resource-group ml-infrastructure-rg \
  --vnet-name ml-vnet \
  --name data-subnet \
  --address-prefix 10.0.3.0/24
```

### Network Security Groups (NSG)

```bash
# Create NSG
az network nsg create \
  --resource-group ml-infrastructure-rg \
  --name ml-training-nsg

# Allow SSH from specific IP
az network nsg rule create \
  --resource-group ml-infrastructure-rg \
  --nsg-name ml-training-nsg \
  --name allow-ssh \
  --priority 100 \
  --source-address-prefixes 1.2.3.4 \
  --destination-port-ranges 22 \
  --protocol Tcp \
  --access Allow

# Allow internal communication
az network nsg rule create \
  --resource-group ml-infrastructure-rg \
  --nsg-name ml-training-nsg \
  --name allow-internal \
  --priority 110 \
  --source-address-prefixes 10.0.0.0/16 \
  --destination-port-ranges "*" \
  --protocol "*" \
  --access Allow
```

### Azure Load Balancer

```bash
# Create public IP
az network public-ip create \
  --resource-group ml-infrastructure-rg \
  --name ml-lb-ip \
  --sku Standard

# Create load balancer
az network lb create \
  --resource-group ml-infrastructure-rg \
  --name ml-load-balancer \
  --sku Standard \
  --public-ip-address ml-lb-ip \
  --frontend-ip-name ml-frontend \
  --backend-pool-name ml-backend-pool

# Create health probe
az network lb probe create \
  --resource-group ml-infrastructure-rg \
  --lb-name ml-load-balancer \
  --name ml-health-probe \
  --protocol http \
  --port 8000 \
  --path /health

# Create load balancing rule
az network lb rule create \
  --resource-group ml-infrastructure-rg \
  --lb-name ml-load-balancer \
  --name ml-http-rule \
  --protocol tcp \
  --frontend-port 80 \
  --backend-port 8000 \
  --frontend-ip-name ml-frontend \
  --backend-pool-name ml-backend-pool \
  --probe-name ml-health-probe
```

---

## Cost Optimization

### 1. Reserved Instances

Save **40-60%** with 1-3 year commitments:

```
┌─────────────────────┬──────────────┬──────────────┬──────────────┐
│ VM Size             │ Pay-as-you-go│ 1-Year RI    │ 3-Year RI    │
├─────────────────────┼──────────────┼──────────────┼──────────────┤
│ Standard_D4s_v3     │ $0.192/hr    │ $0.131/hr    │ $0.086/hr    │
│ Standard_NC6s_v3    │ $3.06/hr     │ $2.08/hr     │ $1.37/hr     │
└─────────────────────┴──────────────┴──────────────┴──────────────┘

Savings:                 -              32%           55%
```

Purchase through Azure Portal: Cost Management → Reservations

### 2. Azure Spot VMs

Save **up to 90%**:

```bash
# Always use Spot VMs for training
az vm create \
  --priority Spot \
  --max-price 0.5 \
  --eviction-policy Deallocate

# Cost comparison:
# Regular NC6s_v3: $3.06/hour
# Spot NC6s_v3: ~$0.30/hour (90% savings)
```

### 3. Auto-shutdown VMs

```bash
# Enable auto-shutdown
az vm auto-shutdown \
  --resource-group ml-infrastructure-rg \
  --name ml-training-vm \
  --time 1900 \
  --timezone "Pacific Standard Time"
```

### 4. Storage Cost Optimization

```bash
# Use lifecycle management
# Hot → Cool (30 days) → Archive (90 days) → Delete (365 days)

# Savings example:
# 1 TB for 1 year:
# - All Hot: $221
# - With lifecycle: $78 (65% savings)
```

### 5. Azure Cost Management

```bash
# View current costs
az consumption usage list --output table

# Create budget
az consumption budget create \
  --budget-name ml-monthly-budget \
  --amount 1000 \
  --time-grain Monthly \
  --time-period start-date=2024-01-01 \
  --notifications threshold=50 threshold-type=Actual contact-emails=["admin@example.com"]

# Set up cost alerts (via Portal)
# Cost Management → Budgets → Create budget
# - Set threshold at 50%, 90%, 100%
# - Email notifications
# - Action groups for automation
```

### Cost Optimization Checklist

- [ ] Use Spot VMs for training (90% savings)
- [ ] Purchase Reserved Instances for production (40-60% savings)
- [ ] Enable VM auto-shutdown (evenings, weekends)
- [ ] Use storage lifecycle management (65% savings)
- [ ] Right-size VMs (avoid over-provisioning)
- [ ] Delete unused resources (disks, IPs, snapshots)
- [ ] Use AKS autoscaling (scale to zero)
- [ ] Leverage Azure Hybrid Benefit (if you have Windows licenses)
- [ ] Set up cost budgets and alerts
- [ ] Review costs weekly with Cost Management

---

## Hands-on Exercise

### Exercise: Deploy Complete ML System on Azure

**Objective**: Deploy an image classification model with:

```
User Request
    ↓
Application Gateway (WAF + SSL)
    ↓
AKS Cluster (autoscaling 2-10 pods)
    ↓
Model Server (loads from Blob Storage)
    ↓
Azure Blob Storage (model weights)
```

**Requirements**:
1. Use Spot VMs for AKS node pool
2. Enable AKS autoscaling (CPU > 70%)
3. Implement health checks
4. Store models in Blob Storage
5. Set up Azure Monitor
6. Estimate monthly cost (1M requests/month)

**Steps**:

1. **Create resources**:
   ```bash
   # Resource group
   az group create --name ml-exercise-rg --location eastus

   # Storage account
   az storage account create --name mlexercisestore --resource-group ml-exercise-rg

   # Upload model
   az storage container create --name models --account-name mlexercisestore
   azcopy copy model.pth "https://mlexercisestore.blob.core.windows.net/models/model.pth?<SAS>"
   ```

2. **Create AKS cluster**:
   ```bash
   az aks create --resource-group ml-exercise-rg --name ml-aks --node-count 2 --node-vm-size Standard_D4s_v3

   # Add Spot node pool
   az aks nodepool add --resource-group ml-exercise-rg --cluster-name ml-aks --name spotpool --priority Spot --eviction-policy Delete --spot-max-price 0.5 --node-count 2
   ```

3. **Deploy model**:
   ```bash
   kubectl apply -f deployment.yaml
   kubectl apply -f service.yaml
   kubectl apply -f hpa.yaml
   ```

4. **Set up Application Gateway**:
   ```bash
   # Follow Azure documentation for Application Gateway + AKS integration
   ```

5. **Configure monitoring**:
   ```bash
   # Enable Container Insights
   az aks enable-addons --resource-group ml-exercise-rg --name ml-aks --addons monitoring
   ```

**Expected Cost** (1M requests/month):
- AKS Spot nodes: 2 × $0.04/hr × 730hr = $58
- Blob Storage: 1 GB × $0.0184 = $0.02
- Application Gateway: ~$150/month
- Monitor: $2.30/GB ingested (~$10/month)
- **Total: ~$220/month**

---

## Self-Check Questions

1. **What's the difference between Azure VM, AKS, and Azure ML for training?**
   <details>
   <summary>Answer</summary>

   - **Azure VM**: Full control, manual management, good for custom setups
   - **AKS**: Container orchestration, good for production serving with auto-scaling
   - **Azure ML**: Fully managed, automated scaling, good for end-to-end ML workflows
   </details>

2. **How does Azure OpenAI differ from OpenAI API?**
   <details>
   <summary>Answer</summary>

   - **Enterprise features**: SLA, security, compliance
   - **Private network**: VNet integration, private endpoints
   - **Pricing**: Different pricing model (per-token)
   - **Regional deployment**: Data residency options
   - **Azure integration**: Works with Azure ML, Key Vault, etc.
   </details>

3. **When should you use Azure Spot VMs?**
   <details>
   <summary>Answer</summary>

   Use Spot VMs for:
   - Training jobs (with checkpointing)
   - Batch processing
   - Non-critical workloads
   - Cost-sensitive projects

   Avoid for:
   - Production inference
   - Time-sensitive workloads
   - Workloads without fault tolerance
   </details>

4. **What's the best storage tier for model checkpoints?**
   <details>
   <summary>Answer</summary>

   - **Active checkpoints**: Hot tier ($0.0184/GB)
   - **Old checkpoints**: Cool tier after 30 days ($0.01/GB)
   - **Archived checkpoints**: Archive tier after 90 days ($0.00099/GB)
   - Use lifecycle management for automatic transition
   </details>

5. **How do you enable autoscaling on AKS?**
   <details>
   <summary>Answer</summary>

   ```bash
   # Cluster autoscaler (nodes)
   az aks update --enable-cluster-autoscaler --min-count 1 --max-count 10

   # Horizontal Pod Autoscaler (pods)
   kubectl autoscale deployment ml-model --cpu-percent=70 --min=2 --max=20
   ```
   </details>

---

## Additional Resources

### Official Documentation
- [Azure ML Documentation](https://docs.microsoft.com/azure/machine-learning/)
- [Azure OpenAI Documentation](https://learn.microsoft.com/azure/cognitive-services/openai/)
- [AKS Documentation](https://docs.microsoft.com/azure/aks/)
- [Azure Blob Storage Documentation](https://docs.microsoft.com/azure/storage/blobs/)

### Tutorials
- [Azure ML Tutorials](https://docs.microsoft.com/azure/machine-learning/tutorial-1st-experiment-sdk-setup)
- [Deploy ML models to AKS](https://docs.microsoft.com/azure/machine-learning/how-to-deploy-azure-kubernetes-service)

### Tools
- [Azure CLI](https://docs.microsoft.com/cli/azure/)
- [Azure Storage Explorer](https://azure.microsoft.com/features/storage-explorer/)
- [Azure ML SDK](https://docs.microsoft.com/python/api/overview/azure/ml/)

### Cost Management
- [Azure Pricing Calculator](https://azure.microsoft.com/pricing/calculator/)
- [Azure Cost Management](https://azure.microsoft.com/services/cost-management/)

---

## Summary

In this lesson, you learned:

✅ Set up Azure account with proper RBAC
✅ Deploy VMs with GPUs for ML training
✅ Use Azure Blob Storage with lifecycle management
✅ Deploy models on AKS with autoscaling
✅ Leverage Azure Machine Learning for managed workflows
✅ Integrate Azure OpenAI (GPT-4, embeddings)
✅ Configure VNet and load balancing
✅ Optimize costs with Spot VMs and Reserved Instances

**Key Takeaways**:
- Azure excels at enterprise ML with strong governance
- Azure OpenAI provides exclusive access to GPT-4
- Spot VMs save up to 90% on training costs
- Azure ML simplifies end-to-end ML workflows
- Strong hybrid cloud capabilities with Azure Arc

**Next Steps**:
- Complete hands-on exercise
- Explore Azure ML Pipelines
- Learn about Azure Arc for hybrid deployments
- Proceed to Lesson 05: Cloud Storage Deep Dive

---

**Estimated Time to Complete**: 6 hours (including hands-on exercise)
**Difficulty**: Intermediate
**Next Lesson**: [05-cloud-storage.md](./05-cloud-storage.md)
