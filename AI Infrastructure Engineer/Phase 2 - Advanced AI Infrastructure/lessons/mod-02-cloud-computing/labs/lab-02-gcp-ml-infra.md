# Lab 02: Provision an ML-Ready GCP Environment

**Duration:** 90 min  **Prerequisites:** GCP project with billing enabled

## Objective
Stand up a minimal ML environment on GCP: a VPC, a GCS bucket for model artifacts, a Compute Engine VM running iris-api, and a global HTTPS load balancer with a managed certificate.

## Steps

### 1. Configure CLI
```bash
gcloud auth login
gcloud config set project <PROJECT_ID>
gcloud config set compute/region us-central1
```

### 2. Enable required APIs
```bash
gcloud services enable compute.googleapis.com storage.googleapis.com artifactregistry.googleapis.com
```

### 3. Create network + bucket
```bash
gcloud compute networks create ml-lab --subnet-mode=auto
gsutil mb -l us-central1 gs://ml-lab-$(gcloud config get-value project)-artifacts
gsutil cp model.joblib gs://ml-lab-$(...)-artifacts/iris/model.joblib
```

### 4. Push iris-api to Artifact Registry
```bash
gcloud artifacts repositories create iris --repository-format=docker --location=us-central1
docker tag iris-api:0.2 us-central1-docker.pkg.dev/$(gcloud config get-value project)/iris/iris-api:0.2
gcloud auth configure-docker us-central1-docker.pkg.dev
docker push us-central1-docker.pkg.dev/$(...)/iris/iris-api:0.2
```

### 5. Create the VM
```bash
gcloud compute instances create-with-container iris-vm \
  --zone=us-central1-a --machine-type=e2-small \
  --container-image=us-central1-docker.pkg.dev/$(...)/iris/iris-api:0.2 \
  --tags=http-server \
  --network=ml-lab
gcloud compute firewall-rules create allow-iris --allow=tcp:80 --target-tags=http-server --network=ml-lab
```

### 6. Front it with a global HTTPS load balancer
- Reserve a global static IP
- Create an unmanaged instance group with the VM
- Create a backend service with HTTP health check `/health`
- Create a URL map → target HTTPS proxy → global forwarding rule
- Attach a Google-managed cert for your domain

### 7. Test
```bash
curl https://<your-domain>/health
```

## Validation
- [ ] HTTPS endpoint returns 200 on `/health` and a valid prediction on `/predict`.
- [ ] Managed cert status is ACTIVE.
- [ ] GCS bucket contains the model artifact.

## Cleanup
```bash
gcloud compute forwarding-rules delete ... && gcloud compute target-https-proxies delete ... && gcloud compute url-maps delete ...
gcloud compute backend-services delete iris-be
gcloud compute instance-groups unmanaged delete iris-ig
gcloud compute instances delete iris-vm --zone=us-central1-a
gcloud artifacts repositories delete iris --location=us-central1
gsutil rm -r gs://ml-lab-$(...)-artifacts
gcloud compute networks delete ml-lab
```

## Troubleshooting
- **Managed cert stuck PROVISIONING > 1 hour** — DNS A record for your domain must point to the LB IP and TTL elapsed.
- **VM container doesn't start** — `gcloud compute instances get-serial-port-output iris-vm --zone=us-central1-a` reveals the docker pull error.
- **APIs disabled error** — re-run the `gcloud services enable` step; propagation can take 30s.
