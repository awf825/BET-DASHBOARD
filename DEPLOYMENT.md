Azure deployment updated · MD
Copy

# Azure Deployment Guide (Updated for Container Registry)

This guide covers deploying the Sports Betting Dashboard to Azure using Azure Container Registry and Container Instances.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Azure Cloud                               │
│                                                                  │
│  ┌──────────────────┐      ┌─────────────────────────────────┐  │
│  │  Azure Container │      │  Azure Blob Storage             │  │
│  │  Instances       │◄────►│  └── betting-data/              │  │
│  │                  │      │      ├── models/                │  │
│  │  Flask Dashboard │      │      │   ├── nhl_model.joblib   │  │
│  │  (Docker)        │      │      │   └── nba_model.joblib   │  │
│  └──────────────────┘      │      └── data/                  │  │
│         ▲                  │          └── nhl_features_latest.csv│
│         │                  └─────────────────────────────────┘  │
│  ┌──────┴───────────┐                    ▲                       │
│  │  Azure Container │                    │                       │
│  │  Registry        │                    │                       │
│  └──────────────────┘                    │                       │
│                                           │                       │
│  ┌──────────────────┐                    │                       │
│  │  Azure Function  │────────────────────┘                      │
│  │  (Timer: 6AM ET) │                                            │
│  │  NHL Scraper     │                                            │
│  └──────────────────┘                                            │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

1. Azure CLI installed: `brew install azure-cli`
2. Azure account with active subscription
3. Python 3.11+
4. Docker Desktop installed (for local testing)

## Step 1: Create Azure Resources

### 1.1 Login to Azure
```bash
az login
```

### 1.2 Create Resource Group
```bash
az group create --name betting-dashboard-rg --location eastus
```

### 1.3 Create Storage Account
```bash
az storage account create \
  --name bettingdashboardstorage \
  --resource-group betting-dashboard-rg \
  --location eastus \
  --sku Standard_LRS
```

### 1.4 Create Blob Container
```bash
az storage container create \
  --name betting-data \
  --account-name bettingdashboardstorage
```

### 1.5 Get Storage Connection String
```bash
az storage account show-connection-string \
  --name bettingdashboardstorage \
  --resource-group betting-dashboard-rg \
  --output tsv
```
Save this connection string - you'll need it for both the Container Instance and Function.

## Step 2: Upload Initial Data

### 2.1 Upload Model Files
```bash
# Get the connection string
CONN_STRING=$(az storage account show-connection-string --name bettingdashboardstorage --resource-group betting-dashboard-rg --output tsv)

# Upload NHL model
az storage blob upload \
  --account-name bettingdashboardstorage \
  --container-name betting-data \
  --name models/nhl_model.joblib \
  --file ./python-nhl-2026/model_logit.joblib \
  --connection-string "$CONN_STRING"

# Upload NBA model
az storage blob upload \
  --account-name bettingdashboardstorage \
  --container-name betting-data \
  --name models/nba_model.joblib \
  --file ./python-nba-2026/Models/homewin_logreg_final.joblib \
  --connection-string "$CONN_STRING"
```

### 2.2 Create Initial Features CSV
You need to export your current team features to CSV format. Create a script or manually export from your SQLite database:

```python
# Run this locally to create initial features CSV
import sqlite3
import pandas as pd

conn = sqlite3.connect('python-nhl-2026/nhl_scrape.sqlite')

# Query to get latest team snapshots (customize based on your schema)
query = """
SELECT DISTINCT
    home_team as team,
    home_goals_for_avg, home_goals_against_avg, home_win_pct,
    away_goals_for_avg, away_goals_against_avg, away_win_pct
FROM games_table_20252026
WHERE date = (SELECT MAX(date) FROM games_table_20252026)
"""
df = pd.read_sql(query, conn)
df.to_csv('nhl_features_latest.csv', index=False)
conn.close()
```

Then upload:
```bash
az storage blob upload \
  --account-name bettingdashboardstorage \
  --container-name betting-data \
  --name data/nhl_features_latest.csv \
  --file ./nhl_features_latest.csv \
  --connection-string "$CONN_STRING"
```

## Step 3: Deploy Flask App with Container Registry

### 3.1 Create Dockerfile in flask-dashboard directory
Create a `Dockerfile` in your `flask-dashboard` directory:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gunicorn

# Copy application code
COPY . .

# Expose port
EXPOSE 5000

# Run with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "600", "--workers", "2", "app:app"]
```

### 3.2 Create Azure Container Registry
```bash
az acr create \
  --name bettingdashboardacr \
  --resource-group betting-dashboard-rg \
  --sku Basic \
  --location eastus \
  --admin-enabled true
```

### 3.3 Build and Push Docker Image
```bash
cd flask-dashboard

# Build and push image directly to ACR
az acr build \
  --registry bettingdashboardacr \
  --image betting-dashboard:latest \
  --file Dockerfile \
  .
```

### 3.4 Get ACR Credentials
```bash
# Store these for the next step
ACR_USERNAME=$(az acr credential show --name bettingdashboardacr --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name bettingdashboardacr --query passwords[0].value -o tsv)
CONN_STRING=$(az storage account show-connection-string --name bettingdashboardstorage --resource-group betting-dashboard-rg --output tsv)
SECRET_KEY=$(openssl rand -hex 32)

echo "ACR Username: $ACR_USERNAME"
echo "ACR Password: $ACR_PASSWORD"
echo "Connection String: $CONN_STRING"
echo "Secret Key: $SECRET_KEY"
```

### 3.5 Deploy to Azure Container Instances
```bash
az container create \
  --name betting-dashboard \
  --resource-group betting-dashboard-rg \
  --image bettingdashboardacr.azurecr.io/betting-dashboard:latest \
  --dns-name-label betting-dashboard-$RANDOM \
  --ports 5000 \
  --cpu 1 \
  --memory 1.5 \
  --registry-login-server bettingdashboardacr.azurecr.io \
  --registry-username $ACR_USERNAME \
  --registry-password $ACR_PASSWORD \
  --environment-variables \
    AZURE_STORAGE_CONNECTION_STRING="$CONN_STRING" \
    AZURE_STORAGE_CONTAINER="betting-data" \
    SECRET_KEY="$SECRET_KEY" \
  --location eastus
```

### 3.6 Get Your App URL
```bash
# Get the FQDN (Fully Qualified Domain Name)
az container show \
  --name betting-dashboard \
  --resource-group betting-dashboard-rg \
  --query ipAddress.fqdn \
  --output tsv
```

Your app will be available at:
```
http://<your-fqdn>:5000
```

Example: `http://betting-dashboard-12345.eastus.azurecontainer.io:5000`

### 3.7 Monitor Container Logs
```bash
# View real-time logs
az container logs \
  --name betting-dashboard \
  --resource-group betting-dashboard-rg \
  --follow

# View container state
az container show \
  --name betting-dashboard \
  --resource-group betting-dashboard-rg \
  --query "{FQDN:ipAddress.fqdn,ProvisioningState:provisioningState,State:instanceView.state}" \
  --output table
```

### 3.8 (Optional) Set up HTTPS with Application Gateway
For production with HTTPS, you'll need Azure Application Gateway (~$125/month). For development, the HTTP endpoint is sufficient.

Alternative: Use Azure Container Apps which includes HTTPS certificates automatically.

## Step 4: Deploy Azure Function (NHL Scraper)

### 4.1 Create Function App
```bash
az functionapp create \
  --name nhl-scraper-func \
  --resource-group betting-dashboard-rg \
  --storage-account bettingdashboardstorage \
  --consumption-plan-location eastus \
  --runtime python \
  --runtime-version 3.11 \
  --functions-version 4 \
  --os-type Linux
```

### 4.2 Configure Function Settings
```bash
az functionapp config appsettings set \
  --name nhl-scraper-func \
  --resource-group betting-dashboard-rg \
  --settings \
    STORAGE_CONTAINER="betting-data"
```

### 4.3 Deploy Function
```bash
cd azure-functions/nhl-scraper

# Install Azure Functions Core Tools if not installed
# brew install azure-functions-core-tools@4

# Deploy
func azure functionapp publish nhl-scraper-func
```

## Step 5: Verify Deployment

### 5.1 Check Container Instance Status
```bash
az container show \
  --name betting-dashboard \
  --resource-group betting-dashboard-rg \
  --query "{FQDN:ipAddress.fqdn,State:instanceView.state,RestartCount:instanceView.currentState.restartCount}" \
  --output table
```

### 5.2 Check Container Logs
```bash
az container logs \
  --name betting-dashboard \
  --resource-group betting-dashboard-rg
```

### 5.3 Check Function Logs
```bash
az functionapp logs read \
  --name nhl-scraper-func \
  --resource-group betting-dashboard-rg
```

### 5.4 Test the Dashboard
Visit `http://<your-fqdn>:5000` and:
1. Click "Run Today's Games" on NHL dashboard
2. Verify games load successfully
3. Check NBA dashboard works (uses nba_api directly)

## Updating Your Application

### Update Container Image
When you make changes to your Flask app:

```bash
cd flask-dashboard

# Rebuild and push
az acr build \
  --registry bettingdashboardacr \
  --image betting-dashboard:latest \
  --file Dockerfile \
  .

# Delete old container
az container delete \
  --name betting-dashboard \
  --resource-group betting-dashboard-rg \
  --yes

# Recreate with new image
az container create \
  --name betting-dashboard \
  --resource-group betting-dashboard-rg \
  --image bettingdashboardacr.azurecr.io/betting-dashboard:latest \
  --dns-name-label betting-dashboard-$RANDOM \
  --ports 5000 \
  --cpu 1 \
  --memory 1.5 \
  --registry-login-server bettingdashboardacr.azurecr.io \
  --registry-username $ACR_USERNAME \
  --registry-password $ACR_PASSWORD \
  --environment-variables \
    AZURE_STORAGE_CONNECTION_STRING="$CONN_STRING" \
    AZURE_STORAGE_CONTAINER="betting-data" \
    SECRET_KEY="$SECRET_KEY" \
  --location eastus
```

## Maintenance

### Manually Trigger Function
```bash
# Trigger the NHL scraper manually
az functionapp function invoke \
  --name nhl-scraper-func \
  --resource-group betting-dashboard-rg \
  --function-name nhl_daily_scraper
```

### Update Model File
```bash
CONN_STRING=$(az storage account show-connection-string --name bettingdashboardstorage --resource-group betting-dashboard-rg --output tsv)

az storage blob upload \
  --account-name bettingdashboardstorage \
  --container-name betting-data \
  --name models/nhl_model.joblib \
  --file ./python-nhl-2026/model_logit.joblib \
  --overwrite \
  --connection-string "$CONN_STRING"
```

### View Blob Storage Contents
```bash
az storage blob list \
  --account-name bettingdashboardstorage \
  --container-name betting-data \
  --output table \
  --connection-string "$CONN_STRING"
```

### Restart Container
```bash
az container restart \
  --name betting-dashboard \
  --resource-group betting-dashboard-rg
```

## Cost Estimate

| Service | SKU | Estimated Monthly Cost |
|---------|-----|------------------------|
| Container Instances | 1 vCPU, 1.5GB RAM | ~$35 (running 24/7) |
| Container Registry | Basic | ~$5 |
| Blob Storage | Standard LRS | ~$1 |
| Functions | Consumption | Free tier (1M executions) |
| **Total** | | **~$41/month** |

**Note**: Container Instances charges by the second. If you only run during certain hours, costs can be reduced significantly.

## Optimization Options

### Option 1: Use Container Apps Instead
For better cost optimization and automatic HTTPS:
```bash
# Create Container Apps environment
az containerapp env create \
  --name betting-dashboard-env \
  --resource-group betting-dashboard-rg \
  --location eastus

# Deploy to Container Apps
az containerapp create \
  --name betting-dashboard-app \
  --resource-group betting-dashboard-rg \
  --environment betting-dashboard-env \
  --image bettingdashboardacr.azurecr.io/betting-dashboard:latest \
  --target-port 5000 \
  --ingress external \
  --registry-server bettingdashboardacr.azurecr.io \
  --registry-username $ACR_USERNAME \
  --registry-password $ACR_PASSWORD \
  --env-vars \
    AZURE_STORAGE_CONNECTION_STRING="$CONN_STRING" \
    AZURE_STORAGE_CONTAINER="betting-data" \
    SECRET_KEY="$SECRET_KEY" \
  --cpu 0.5 \
  --memory 1Gi \
  --min-replicas 0 \
  --max-replicas 1
```

Container Apps benefits:
- Automatic HTTPS with custom domains
- Scale to zero when not in use (lower costs)
- Automatic SSL certificate management
- Better for production workloads

### Option 2: Scheduled Start/Stop
Save costs by stopping the container when not needed:
```bash
# Stop container
az container stop --name betting-dashboard --resource-group betting-dashboard-rg

# Start container
az container start --name betting-dashboard --resource-group betting-dashboard-rg
```

## Local Development

The app automatically detects local vs production mode:

- **Local**: No `AZURE_STORAGE_CONNECTION_STRING` set → uses local SQLite/files
- **Production**: `AZURE_STORAGE_CONNECTION_STRING` set → uses Azure Blob Storage

To test production mode locally:
```bash
export AZURE_STORAGE_CONNECTION_STRING="your_connection_string"
python -m flask run
```

To test the Docker container locally:
```bash
cd flask-dashboard

# Build locally
docker build -t betting-dashboard:test .

# Run locally
docker run -p 5000:5000 \
  -e AZURE_STORAGE_CONNECTION_STRING="your_connection_string" \
  -e AZURE_STORAGE_CONTAINER="betting-data" \
  -e SECRET_KEY="test-secret-key" \
  betting-dashboard:test
```

## Troubleshooting

### Container fails to start
Check logs:
```bash
az container logs --name betting-dashboard --resource-group betting-dashboard-rg
```

Common issues:
- Missing dependencies in requirements.txt
- Incorrect environment variables
- Port configuration mismatch

### "NHL model not found"
- Verify model is uploaded: 
  ```bash
  az storage blob list --container-name betting-data --prefix models/ --account-name bettingdashboardstorage
  ```
- Check connection string is set correctly in container environment variables

### Function not running
- Check timer schedule (runs at 6 AM ET / 11 AM UTC)
- View function logs for errors
- Manually trigger to test

### Container Registry authentication fails
- Verify admin is enabled:
  ```bash
  az acr update --name bettingdashboardacr --admin-enabled true
  ```
- Regenerate credentials if needed:
  ```bash
  az acr credential renew --name bettingdashboardacr --password-name password
  ```

### Slow initial load
- Container Instances have no cold start delay
- If still slow, increase CPU/memory allocation
- Consider using Container Apps for better performance

## Migration from App Service

If you previously deployed to App Service and want to migrate:

1. Export your data from App Service
2. Follow steps 3.1-3.6 above
3. Update DNS/custom domains to point to new Container Instance
4. Delete old App Service resources once verified

## Security Considerations

1. **Secrets Management**: Consider using Azure Key Vault for storing connection strings
2. **Network Security**: For production, use Virtual Network integration
3. **HTTPS**: Use Container Apps or Application Gateway for HTTPS
4. **Container Registry**: Keep admin access disabled unless needed, use managed identities instead