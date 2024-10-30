Write-Host "Setting up your Azure environment configuration..."

# Retrieve Subscription ID and Name dynamically, with user confirmation
$subscriptionDetails = az account show --query "{id:id, name:name}" -o json | ConvertFrom-Json
$subscriptionId = $subscriptionDetails.id
$subscriptionName = $subscriptionDetails.name

Write-Host "Detected Subscription: $subscriptionName (ID: $subscriptionId)"
$confirmSubscription = Read-Host "Would you like to use this subscription? (Y/N)"
if ($confirmSubscription -eq "N") {
    $subscriptionId = Read-Host "Please enter the Subscription ID you'd like to use"
    $subscriptionName = Read-Host "Please enter the Subscription Name for reference"
}
[Environment]::SetEnvironmentVariable("AZURE_SUBSCRIPTION_ID", $subscriptionId)
[Environment]::SetEnvironmentVariable("AZURE_SUBSCRIPTION_NAME", $subscriptionName)
Write-Host "Subscription set to: $subscriptionName (ID: $subscriptionId)"

# Prompt for Resource Group name if not set
$resourceGroup = Read-Host "Enter the name of your Resource Group"
[Environment]::SetEnvironmentVariable("AZURE_RESOURCE_GROUP", $resourceGroup)
Write-Host "Resource Group set to: $resourceGroup"

# Prompt for Location (Region)
$location = Read-Host "Enter the location (e.g., eastus, westeurope) for your resources"
[Environment]::SetEnvironmentVariable("AZURE_LOCATION", $location)
Write-Host "Location set to: $location"

Write-Host "Azure environment variables have been configured successfully."
