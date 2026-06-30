variable "project" {
  type        = string
  description = "GCP project id"
}

variable "region" {
  type    = string
  default = "us-central1"
}

variable "name" {
  type    = string
  default = "mixle-mlops"
}

variable "bucket_name" {
  type        = string
  description = "Globally-unique GCS bucket name for the object store"
}

variable "network" {
  type        = string
  description = "Existing VPC self-link/id for private Cloud SQL + Memorystore"
}

variable "gke_cluster_name" {
  type        = string
  description = "Existing GKE cluster to deploy the Helm chart into (optional reference)"
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

variable "db_tier" {
  type    = string
  default = "db-f1-micro"
}

variable "postgres_version" {
  type    = string
  default = "POSTGRES_16"
}

variable "redis_memory_gb" {
  type    = number
  default = 1
}

variable "labels" {
  type    = map(string)
  default = { app = "mixle-mlops" }
}
