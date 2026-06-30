variable "resource_group" {
  type        = string
  description = "Resource group name (created if create_resource_group=true, else must exist)"
}

variable "create_resource_group" {
  type    = bool
  default = true
}

variable "location" {
  type    = string
  default = "eastus"
}

variable "name" {
  type    = string
  default = "mixle-mlops"
}

variable "storage_account_name" {
  type        = string
  description = "Globally-unique storage account name (3-24 lowercase alphanumerics)"
}

variable "container_name" {
  type    = string
  default = "mixle-objects"
}

variable "aks_cluster_name" {
  type        = string
  description = "Existing AKS cluster to deploy the Helm chart into (optional reference)"
  default     = ""
}

variable "db_username" {
  type    = string
  default = "mixle"
}

variable "db_password" {
  type      = string
  sensitive = true
}

variable "db_sku" {
  type    = string
  default = "B_Standard_B1ms"
}

variable "postgres_version" {
  type    = string
  default = "16"
}

variable "redis_capacity" {
  type    = number
  default = 0
}

variable "tags" {
  type    = map(string)
  default = { app = "mixle-mlops" }
}
