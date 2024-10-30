targetScope = 'subscription'

// Parameters for deployment
@minLength(1)
@maxLength(64)
@description('Name of the environment used to generate a short unique hash for resources.')
param environmentName string

@minLength(1)
@description('Primary location for all resources')
@metadata({
  azd: {
    type: 'location'
  }
})
param location string

@description('The name of an existing resource group where all resources will be deployed.')
param resourceGroupName string

// Optional OpenAI configuration
param openAiServiceName string = 'openai-${uniqueString(environmentName, subscription().id)}'
param openAiSkuName string = 'S0'
param openAiHost string = 'azure'

// Generate a unique token for use in resource names
var resourceToken = toLower(uniqueString(subscription().id, environmentName, location))

// Use an existing resource group
resource rg 'Microsoft.Resources/resourceGroups@2024-03-01' existing = {
  name: resourceGroupName
}

/// TODO: MAKE TAGS APPLY TO ALL RESOURCES
// Deploy the OpenAI service within the existing resource group
module openAi 'core/ai/cognitiveservices.bicep' = if (openAiHost == 'azure') {
  name: 'openai'
  scope: rg
  params: {
    name: openAiServiceName
    location: location
    tags: {
      'az-env-name': environmentName
      'cost-center': '073119414700'
      'segment': 'ee'
      'environment': 'test'
    }
    sku: {
      name: openAiSkuName
    }
    deployments: [
      {
        name: 'chatgpt-${resourceToken}'
        model: {
          format: 'OpenAI'
          name: 'gpt-4o'
        }
        sku: {
          name: 'Standard'
          capacity: 50
        }
        dynamicThrottlingEnabled: true
      }
      {
        name: 'embedding-${resourceToken}'
        model: {
          format: 'OpenAI'
          name: 'text-embedding-3-large'
        }
        sku: {
          name: 'Standard'
          capacity: 50
        }
        dynamicThrottlingEnabled: true
      }
    ]
  }
}

// Deploy core infrastructure (e.g., Blob Storage, Azure Search)
module infra 'azuredeploy.bicep' = {
  name: 'infra'
  scope: rg
  dependsOn: [
    openAi
  ]
  params: {
    location: location
    azureSearchName: 'search-${resourceToken}'
    cognitiveServiceName: 'cognitive-service-${resourceToken}'
    blobStorageAccountName: 'blobstorage${resourceToken}'
  }
}

// Deploy frontend resources with dependency on core infrastructure and OpenAI
module frontend '../apps/frontend/azuredeploy-frontend.bicep' = {
  name: 'frontend'
  scope: rg
  params: {
    appServicePlanSKU: 'S3'
    location: location
    azureOpenAIName: openAi.outputs.name
    azureOpenAIAPIKey: openAi.outputs.key
    azureOpenAIModelName: 'gpt-4o'
    azureOpenAIAPIVersion: '2024-05-13'
    azureSearchName: infra.outputs.azureSearchName
    blobSASToken: 'your-generated-sas-token' // Replace with the actual SAS token
    resourceGroupSearch: rg.name
  }
  dependsOn: [
    openAi
    infra
  ]
}

// Outputs for reference
output AZURE_RESOURCE_GROUP string = rg.name
output AZURE_OPENAI_SERVICE string = openAi.outputs.name
output AZURE_FRONTEND_WEBAPP_NAME string = frontend.outputs.webAppName
output AZURE_SEARCH_ENDPOINT string = infra.outputs.azureSearchEndpoint
output AZURE_BLOB_STORAGE_ACCOUNT_NAME string = infra.outputs.blobStorageAccountName
#disable-next-line outputs-should-not-contain-secrets
