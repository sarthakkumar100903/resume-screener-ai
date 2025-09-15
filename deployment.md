# ☁️ Azure Deployment Guide

**Simple step-by-step guide to deploy on Microsoft Azure**

## 🎯 What You Need

### Azure Services Required:
1. **Azure OpenAI** - For AI analysis
2. **Azure Storage** - For file storage  
3. **Azure Container Instance** - For hosting the app

### Prerequisites:
- Azure subscription ($200 free credit for new users)
- Azure OpenAI access (requires approval - can take 1-2 weeks)
- Domain name (optional)

## 🚀 Deployment Steps

### Step 1: Create Azure Resources

#### 1.1 Create Resource Group
```bash
az login
az group create --name "rg-resume-screener" --location "East US"
```

#### 1.2 Create OpenAI Service
```bash
az cognitiveservices account create \
  --name "openai-resume-screener" \
  --resource-group "rg-resume-screener" \
  --location "East US" \
  --kind "OpenAI" \
  --sku "S0"
```

#### 1.3 Deploy AI Models
```bash
# Deploy GPT-4
az cognitiveservices account deployment create \
  --name "openai-resume-screener" \
  --resource-group "rg-resume-screener" \
  --deployment-name "gpt-4" \
  --model-name "gpt-4" \
  --scale-settings-capacity 10

# Deploy Embedding Model  
az cognitiveservices account deployment create \
  --name "openai-resume-screener" \
  --resource-group "rg-resume-screener" \
  --deployment-name "embedding" \
  --model-name "text-embedding-ada-002" \
  --scale-settings-capacity 10
```

#### 1.4 Create Storage Account
```bash
# Create storage
az storage account create \
  --name "stresumefiles123" \
  --resource-group "rg-resume-screener" \
  --sku "Standard_LRS"

# Create containers
az storage container create --name "resumes" --account-name "stresumefiles123"
az storage container create --name "summaries" --account-name "stresumefiles123"  
az storage container create --name "csv-exports" --account-name "stresumefiles123"
```

### Step 2: Get Credentials

#### 2.1 OpenAI Credentials
```bash
# Get API Key
az cognitiveservices account keys list \
  --name "openai-resume-screener" \
  --resource-group "rg-resume-screener"

# Get Endpoint
az cognitiveservices account show \
  --name "openai-resume-screener" \
  --resource-group "rg-resume-screener" \
  --query "properties.endpoint"
```

#### 2.2 Storage Connection String
```bash
az storage account show-connection-string \
  --name "stresumefiles123" \
  --resource-group "rg-resume-screener"
```

### Step 3: Deploy Application

#### 3.1 Create Dockerfile
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

#### 3.2 Build and Push to Azure
```bash
# Create container registry
az acr create --name "acresumeapp" --resource-group "rg-resume-screener" --sku "Basic"

# Build and push
az acr build --registry "acresumeapp" --image "resume-screener:v1" .
```

#### 3.3 Deploy Container
```bash
az container create \
  --resource-group "rg-resume-screener" \
  --name "resume-screener-app" \
  --image "acresumeapp.azurecr.io/resume-screener:v1" \
  --ports 8501 \
  --dns-name-label "my-resume-screener" \
  --cpu 2 \
  --memory 4 \
  --environment-variables \
    AZURE_OPENAI_KEY="your-openai-key-here" \
    AZURE_OPENAI_ENDPOINT="your-openai-endpoint-here" \
    AZURE_STORAGE_CONNECTION_STRING="your-storage-connection-string"
```

### Step 4: Access Your App
Your app will be available at:
`https://my-resume-screener.eastus.azurecontainer.io:8501`

## 🔧 Configuration Files

### docker-compose.yml (for local testing)
```yaml
version: '3.8'
services:
  resume-screener:
    build: .
    ports:
      - "8501:8501"
    environment:
      - AZURE_OPENAI_KEY=${AZURE_OPENAI_KEY}
      - AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT}
      - AZURE_STORAGE_CONNECTION_STRING=${AZURE_STORAGE_CONNECTION_STRING}
      - HR_EMAIL=${HR_EMAIL}
      - SMTP_PASS=${SMTP_PASS}
```

### .env.production
```bash
# Azure OpenAI
AZURE_OPENAI_KEY=your_production_key_here
AZURE_OPENAI_ENDPOINT=https://your-prod-endpoint.openai.azure.com/

# Azure Storage  
AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=..."

# Email Settings
HR_EMAIL=hr@yourcompany.com
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=hr@yourcompany.com
SMTP_PASS=your_gmail_app_password

# Performance Settings
MAX_RESUME_CHUNKS=3
BATCH_SIZE=5
REQUEST_TIMEOUT=30
```

## 💰 Cost Breakdown

### Monthly Costs (USD):
- **OpenAI API**: $10-200 (depends on usage)
- **Storage**: $1-5 (depends on file count)
- **Container Instance**: $15-50 (depends on size)
- **Total**: $30-300/month

### Cost Optimization Tips:
- Use smaller container sizes for testing
- Set up auto-scaling for production
- Monitor API usage regularly
- Archive old files to cheaper storage tiers

## 🔍 Troubleshooting

### Common Issues:

#### "OpenAI Access Denied"
- Check if you have Azure OpenAI access approved
- Verify API key and endpoint are correct
- Ensure models are deployed properly

#### "Storage Connection Failed"  
- Verify connection string format
- Check if containers exist
- Ensure storage account is accessible

#### "Container Won't Start"
- Check container logs: `az container logs --name "resume-screener-app" --resource-group "rg-resume-screener"`
- Verify all environment variables are set
- Check if ports are configured correctly

#### "Slow Processing"
- Increase container CPU/memory
- Check OpenAI API quotas
- Monitor network connectivity

### Quick Fixes:
```bash
# Restart container
az container restart --name "resume-screener-app" --resource-group "rg-resume-screener"

# Check container status
az container show --name "resume-screener-app" --resource-group "rg-resume-screener"

# View logs
az container logs --name "resume-screener-app" --resource-group "rg-resume-screener"
```

## 🛡️ Security Checklist

- ✅ Use Azure Key Vault for production secrets
- ✅ Enable HTTPS with custom domain
- ✅ Set up firewall rules for storage
- ✅ Use managed identities instead of connection strings
- ✅ Enable audit logging
- ✅ Regular backup of data

## 📊 Monitoring Setup

### Basic Health Check:
```bash
# Check if app is responding
curl https://my-resume-screener.eastus.azurecontainer.io:8501/_stcore/health
```

### Application Insights (Optional):
```bash
# Create monitoring
az monitor app-insights component create \
  --app "resume-screener-insights" \
  --location "East US" \
  --resource-group "rg-resume-screener"
```

## 🚀 Scaling for Production

### Option 1: Multiple Container Instances
```bash
# Create additional instances for load balancing
az container create \
  --name "resume-screener-app-2" \
  --resource-group "rg-resume-screener" \
  --image "acresumeapp.azurecr.io/resume-screener:v1" \
  # ... same config as main container
```

### Option 2: Azure App Service
```bash
# More scalable option for high traffic
az webapp create \
  --resource-group "rg-resume-screener" \
  --plan "my-app-plan" \
  --name "resume-screener-webapp" \
  --deployment-container-image-name "acresumeapp.azurecr.io/resume-screener:v1"
```
