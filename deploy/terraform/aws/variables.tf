variable "region" {
  type        = string
  description = "AWS region"
  default     = "us-east-1"
}

variable "name" {
  type        = string
  description = "Name prefix for created resources"
  default     = "mixle-mlops"
}

variable "bucket_name" {
  type        = string
  description = "Globally-unique S3 bucket name for the object store"
}

variable "vpc_id" {
  type        = string
  description = "Existing VPC id to place RDS/ElastiCache in"
}

variable "subnet_ids" {
  type        = list(string)
  description = "Existing (private) subnet ids for RDS/ElastiCache"
}

variable "allowed_cidrs" {
  type        = list(string)
  description = "CIDRs allowed to reach Postgres/Redis (e.g. your EKS node/pod CIDRs)"
  default     = ["10.0.0.0/8"]
}

variable "eks_cluster_name" {
  type        = string
  description = "Existing EKS cluster name to deploy the Helm chart into (optional reference)"
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

variable "db_instance_class" {
  type    = string
  default = "db.t4g.micro"
}

variable "postgres_version" {
  type    = string
  default = "16"
}

variable "redis_node_type" {
  type    = string
  default = "cache.t4g.micro"
}

variable "tags" {
  type    = map(string)
  default = { app = "mixle-mlops" }
}
