# Azure Deployment Guide

This guide covers deploying the Sports Betting Dashboard to Azure.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Azure Cloud                               │
│                                                                  │
│  ┌──────────────────┐      ┌─────────────────────────────────┐  │
│  │  Azure App       │      │  Azure Blob Storage             │  │
│  │  Service (B1)    │◄────►│  └── betting-data/              │  │
│  │                  │      │      ├── models/                │  │
│  │  Flask Dashboard │      │      │   ├── nhl_model.joblib   │  │
│  │                  │      │      │   └── nba_model.joblib   │  │
│  └──────────────────┘      │      └── data/                  │  │
│                            │          └── nhl_features_latest.csv│
│  ┌──────────────────┐      └─────────────────────────────────┘  │
│  │  Azure Function  │                    ▲                       │
│  │  (Timer: 6AM ET) │────────────────────┘                      │
│  │  NHL Scraper     │                                            │
│  └──────────────────┘                                            │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

1. Azure CLI installed: `brew install azure-cli`
2. Azure account with active subscription
3. Python 3.11+

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
Save this connection string - you'll need it for both the App Service and Function.

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

## Step 3: Deploy Flask App to Azure App Service

### 3.1 Create App Service Plan
```bash
az appservice plan create \
  --name betting-dashboard-plan \
  --resource-group betting-dashboard-rg \
  --sku B1 \
  --is-linux
```

### 3.2 Create Web App
```bash
az webapp create \
  --name betting-dashboard-app \
  --resource-group betting-dashboard-rg \
  --plan betting-dashboard-plan \
  --runtime "PYTHON:3.11"
```

### 3.3 Configure Environment Variables
```bash
# Get your storage connection string
CONN_STRING=$(az storage account show-connection-string --name bettingdashboardstorage --resource-group betting-dashboard-rg --output tsv)

az webapp config appsettings set \
  --name betting-dashboard-app \
  --resource-group betting-dashboard-rg \
  --settings \
    AZURE_STORAGE_CONNECTION_STRING="$CONN_STRING" \
    AZURE_STORAGE_CONTAINER="betting-data" \
    SECRET_KEY="$(openssl rand -hex 32)"
```

### 3.4 Deploy Flask App
```bash
cd flask-dashboard

# Create deployment package
zip -r ../deploy.zip . -x "*.pyc" -x "__pycache__/*" -x "venv/*" -x ".git/*"

# Deploy
az webapp deploy \
  --name betting-dashboard-app \
  --resource-group betting-dashboard-rg \
  --src-path ../deploy.zip \
  --type zip
```

### 3.5 Configure Startup Command
```bash
az webapp config set \
  --name betting-dashboard-app \
  --resource-group betting-dashboard-rg \
  --startup-file "gunicorn --bind=0.0.0.0 --timeout 600 app:app"
```

### 3.6 Access Your App
Your app will be available at:
```
https://betting-dashboard-app.azurewebsites.net
```

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

### 5.1 Check App Service Logs
```bash
az webapp log tail \
  --name betting-dashboard-app \
  --resource-group betting-dashboard-rg
```

### 5.2 Check Function Logs
```bash
az functionapp logs read \
  --name nhl-scraper-func \
  --resource-group betting-dashboard-rg
```

### 5.3 Test the Dashboard
Visit `https://betting-dashboard-app.azurewebsites.net` and:
1. Click "Run Today's Games" on NHL dashboard
2. Verify games load successfully
3. Check NBA dashboard works (uses nba_api directly)

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

## Cost Estimate

| Service | SKU | Estimated Monthly Cost |
|---------|-----|------------------------|
| App Service | B1 | ~$13 |
| Blob Storage | Standard LRS | ~$1 |
| Functions | Consumption | Free tier (1M executions) |
| **Total** | | **~$14/month** |

## Local Development

The app automatically detects local vs production mode:

- **Local**: No `AZURE_STORAGE_CONNECTION_STRING` set → uses local SQLite/files
- **Production**: `AZURE_STORAGE_CONNECTION_STRING` set → uses Azure Blob Storage

To test production mode locally:
```bash
export AZURE_STORAGE_CONNECTION_STRING="your_connection_string"
python -m flask run
```

## Troubleshooting

### App shows "NHL model not found"
- Verify model is uploaded: `az storage blob list --container-name betting-data --prefix models/`
- Check connection string is set correctly in App Service settings

### Function not running
- Check timer schedule (runs at 6 AM ET / 11 AM UTC)
- View function logs for errors
- Manually trigger to test

### Slow initial load
- First request after idle may take 10-30 seconds (cold start)
- Consider upgrading to B2 or enabling "Always On" for faster response
