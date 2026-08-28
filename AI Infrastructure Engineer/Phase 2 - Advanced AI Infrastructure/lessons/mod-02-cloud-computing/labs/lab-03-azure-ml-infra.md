# Lab 03: Provision an ML-Ready Azure Environment

**Duration:** 90 min  **Prerequisites:** Azure subscription with Contributor

## Objective
Provision a minimal Azure ML environment: a resource group, a virtual network, a storage account for model artifacts, an Azure Container Registry, and an Azure Container App running iris-api with a public ingress.

## Steps

### 1. Configure CLI
```bash
az login
az account set --subscription "<NAME-OR-ID>"
LOC=eastus
RG=ml-lab-rg
az group create -n $RG -l $LOC
```

### 2. Storage + container for artifacts
```bash
SA=mllabsa$RANDOM
az storage account create -n $SA -g $RG -l $LOC --sku Standard_LRS
az storage container create -n artifacts --account-name $SA
az storage blob upload --account-name $SA -c artifacts -n iris/model.joblib -f model.joblib
```

### 3. Container Registry + push image
```bash
ACR=mllabacr$RANDOM
az acr create -n $ACR -g $RG --sku Basic --admin-enabled true
az acr login -n $ACR
docker tag iris-api:0.2 $ACR.azurecr.io/iris-api:0.2
docker push $ACR.azurecr.io/iris-api:0.2
```

### 4. Container Apps environment + app
```bash
az extension add -n containerapp --upgrade
az provider register -n Microsoft.App --wait

ENV=ml-lab-env
az containerapp env create -n $ENV -g $RG -l $LOC

APP=iris-api
az containerapp create -n $APP -g $RG --environment $ENV \
  --image $ACR.azurecr.io/iris-api:0.2 \
  --target-port 8000 --ingress external \
  --registry-server $ACR.azurecr.io \
  --query properties.configuration.ingress.fqdn -o tsv
```

### 5. Test
```bash
FQDN=$(az containerapp show -n $APP -g $RG --query properties.configuration.ingress.fqdn -o tsv)
curl https://$FQDN/health
curl -X POST https://$FQDN/predict -H 'content-type: application/json' \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

## Validation
- [ ] HTTPS endpoint returns 200 on `/health`.
- [ ] `/predict` returns a JSON prediction.
- [ ] Storage account contains `iris/model.joblib` blob.
- [ ] ACR contains the `iris-api:0.2` image.

## Cleanup
```bash
az group delete -n $RG --yes --no-wait
```

## Troubleshooting
- **`az containerapp create` fails with provider not registered** — `az provider register -n Microsoft.App --wait` and retry.
- **Image pull fails** — Container App must have registry credentials. The `--registry-server` flag also sets identity credentials when admin is enabled on ACR.
- **Region not supported** — Container Apps is GA in most regions but not all; try `eastus`, `westeurope`, or `australiaeast`.
