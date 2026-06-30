# Minimal AWS module for mixle-mlops managed dependencies: an S3 bucket (object store), an RDS Postgres
# instance, and an ElastiCache Redis cluster. It references an EXISTING VPC/subnets and an EXISTING EKS
# cluster (so you can layer it onto your platform) and outputs exactly the env the Helm chart needs.
#
# This is intentionally small and opinionated for a quickstart, not a production-hardened module.
# terraform/docker are not required to read it; `terraform init && terraform apply` provisions it.

terraform {
  required_version = ">= 1.3"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

# --- object store: S3 bucket ------------------------------------------------
resource "aws_s3_bucket" "objects" {
  bucket = var.bucket_name
  tags   = var.tags
}

resource "aws_s3_bucket_versioning" "objects" {
  bucket = aws_s3_bucket.objects.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "objects" {
  bucket                  = aws_s3_bucket.objects.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# --- managed Postgres: RDS --------------------------------------------------
resource "aws_db_subnet_group" "pg" {
  name       = "${var.name}-pg"
  subnet_ids = var.subnet_ids
  tags       = var.tags
}

resource "aws_security_group" "pg" {
  name   = "${var.name}-pg"
  vpc_id = var.vpc_id
  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidrs
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = var.tags
}

resource "aws_db_instance" "pg" {
  identifier             = "${var.name}-pg"
  engine                 = "postgres"
  engine_version         = var.postgres_version
  instance_class         = var.db_instance_class
  allocated_storage      = 20
  db_name                = "mixle"
  username               = var.db_username
  password               = var.db_password
  db_subnet_group_name   = aws_db_subnet_group.pg.name
  vpc_security_group_ids = [aws_security_group.pg.id]
  skip_final_snapshot    = true
  publicly_accessible    = false
  tags                   = var.tags
}

# --- managed Redis: ElastiCache --------------------------------------------
resource "aws_elasticache_subnet_group" "redis" {
  name       = "${var.name}-redis"
  subnet_ids = var.subnet_ids
}

resource "aws_security_group" "redis" {
  name   = "${var.name}-redis"
  vpc_id = var.vpc_id
  ingress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidrs
  }
  tags = var.tags
}

resource "aws_elasticache_cluster" "redis" {
  cluster_id           = "${var.name}-redis"
  engine               = "redis"
  node_type            = var.redis_node_type
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  subnet_group_name    = aws_elasticache_subnet_group.redis.name
  security_group_ids   = [aws_security_group.redis.id]
  tags                 = var.tags
}

# --- (reference) EKS cluster the Helm chart deploys into --------------------
data "aws_eks_cluster" "this" {
  count = var.eks_cluster_name == "" ? 0 : 1
  name  = var.eks_cluster_name
}
