# Minimal Alibaba Cloud module for mixle-mlops managed dependencies: an OSS bucket (object store), an ApsaraDB
# RDS PostgreSQL instance, and an ApsaraDB for Redis (KVStore) instance. Outputs the env the Helm chart needs.
# References an existing ACK cluster (optional). Small/opinionated quickstart, not production-hardened.

terraform {
  required_version = ">= 1.3"
  required_providers {
    alicloud = {
      source  = "aliyun/alicloud"
      version = ">= 1.200"
    }
  }
}

provider "alicloud" {
  region = var.region
}

# --- object store: OSS bucket ----------------------------------------------
resource "alicloud_oss_bucket" "objects" {
  bucket = var.bucket_name
  acl    = "private"
  versioning {
    status = "Enabled"
  }
  tags = var.tags
}

# --- managed Postgres: ApsaraDB RDS ----------------------------------------
resource "alicloud_db_instance" "pg" {
  engine               = "PostgreSQL"
  engine_version       = var.postgres_version
  instance_type        = var.db_instance_type
  instance_storage     = 20
  instance_name        = "${var.name}-pg"
  vswitch_id           = var.vswitch_id
  security_ips         = var.allowed_ips
  db_instance_storage_type = "cloud_essd"
  tags                 = var.tags
}

resource "alicloud_db_database" "mixle" {
  instance_id = alicloud_db_instance.pg.id
  name        = "mixle"
}

resource "alicloud_rds_account" "mixle" {
  db_instance_id   = alicloud_db_instance.pg.id
  account_name     = var.db_username
  account_password = var.db_password
}

# --- managed Redis: ApsaraDB for Redis (KVStore) ---------------------------
resource "alicloud_kvstore_instance" "redis" {
  db_instance_name = "${var.name}-redis"
  instance_class   = var.redis_instance_class
  instance_type    = "Redis"
  engine_version   = var.redis_version
  vswitch_id       = var.vswitch_id
  password         = var.redis_password
  security_ips     = var.allowed_ips
  tags             = var.tags
}

# --- (reference) ACK cluster the Helm chart deploys into -------------------
data "alicloud_cs_managed_kubernetes_clusters" "this" {
  count       = var.ack_cluster_name == "" ? 0 : 1
  name_regex  = var.ack_cluster_name
}
