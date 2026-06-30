# Minimal GCP module for mixle-mlops managed dependencies: a Cloud Storage bucket (object store), a Cloud SQL
# Postgres instance, and a Memorystore Redis instance. Outputs the env the Helm chart needs. References an
# existing GKE cluster (optional). Small/opinionated quickstart, not production-hardened.

terraform {
  required_version = ">= 1.3"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.0"
    }
  }
}

provider "google" {
  project = var.project
  region  = var.region
}

# --- object store: Cloud Storage bucket ------------------------------------
resource "google_storage_bucket" "objects" {
  name                        = var.bucket_name
  location                    = var.region
  uniform_bucket_level_access = true
  versioning {
    enabled = true
  }
  labels = var.labels
}

# --- managed Postgres: Cloud SQL -------------------------------------------
resource "google_sql_database_instance" "pg" {
  name             = "${var.name}-pg"
  database_version = var.postgres_version
  region           = var.region
  settings {
    tier = var.db_tier
    ip_configuration {
      ipv4_enabled    = false
      private_network = var.network
    }
  }
  deletion_protection = false
}

resource "google_sql_database" "mixle" {
  name     = "mixle"
  instance = google_sql_database_instance.pg.name
}

resource "google_sql_user" "mixle" {
  name     = var.db_username
  instance = google_sql_database_instance.pg.name
  password = var.db_password
}

# --- managed Redis: Memorystore -------------------------------------------
resource "google_redis_instance" "redis" {
  name               = "${var.name}-redis"
  tier               = "BASIC"
  memory_size_gb     = var.redis_memory_gb
  region             = var.region
  authorized_network = var.network
  labels             = var.labels
}

# --- (reference) GKE cluster the Helm chart deploys into -------------------
data "google_container_cluster" "this" {
  count    = var.gke_cluster_name == "" ? 0 : 1
  name     = var.gke_cluster_name
  location = var.region
}
