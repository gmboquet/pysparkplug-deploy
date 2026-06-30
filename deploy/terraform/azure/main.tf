# Minimal Azure module for mixle-mlops managed dependencies: a Storage Account + Blob container (object store),
# an Azure Database for PostgreSQL Flexible Server, and an Azure Cache for Redis. Outputs the env the Helm chart
# needs. References an existing AKS cluster (optional). Small/opinionated quickstart, not production-hardened.

terraform {
  required_version = ">= 1.3"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = ">= 3.80"
    }
  }
}

provider "azurerm" {
  features {}
}

resource "azurerm_resource_group" "this" {
  count    = var.create_resource_group ? 1 : 0
  name     = var.resource_group
  location = var.location
}

locals {
  rg_name     = var.create_resource_group ? azurerm_resource_group.this[0].name : var.resource_group
  rg_location = var.location
}

# --- object store: Storage Account + container -----------------------------
resource "azurerm_storage_account" "objects" {
  name                     = var.storage_account_name
  resource_group_name      = local.rg_name
  location                 = local.rg_location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  min_tls_version          = "TLS1_2"
  tags                     = var.tags
}

resource "azurerm_storage_container" "objects" {
  name                  = var.container_name
  storage_account_name  = azurerm_storage_account.objects.name
  container_access_type = "private"
}

# --- managed Postgres: Flexible Server -------------------------------------
resource "azurerm_postgresql_flexible_server" "pg" {
  name                          = "${var.name}-pg"
  resource_group_name           = local.rg_name
  location                      = local.rg_location
  version                       = var.postgres_version
  administrator_login           = var.db_username
  administrator_password        = var.db_password
  storage_mb                    = 32768
  sku_name                      = var.db_sku
  public_network_access_enabled = true
  zone                          = "1"
  tags                          = var.tags
}

resource "azurerm_postgresql_flexible_server_database" "mixle" {
  name      = "mixle"
  server_id = azurerm_postgresql_flexible_server.pg.id
  charset   = "UTF8"
  collation = "en_US.utf8"
}

# --- managed Redis: Azure Cache for Redis ----------------------------------
resource "azurerm_redis_cache" "redis" {
  name                 = "${var.name}-redis"
  resource_group_name  = local.rg_name
  location             = local.rg_location
  capacity             = var.redis_capacity
  family               = "C"
  sku_name             = "Basic"
  non_ssl_port_enabled = false
  minimum_tls_version  = "1.2"
  tags                 = var.tags
}

# --- (reference) AKS cluster the Helm chart deploys into -------------------
data "azurerm_kubernetes_cluster" "this" {
  count               = var.aks_cluster_name == "" ? 0 : 1
  name                = var.aks_cluster_name
  resource_group_name = local.rg_name
}
