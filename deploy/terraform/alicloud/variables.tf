variable "region" {
  type    = string
  default = "cn-hangzhou"
}

variable "name" {
  type    = string
  default = "mixle-mlops"
}

variable "bucket_name" {
  type        = string
  description = "Globally-unique OSS bucket name for the object store"
}

variable "vswitch_id" {
  type        = string
  description = "Existing VSwitch id (in your VPC) for RDS/KVStore"
}

variable "allowed_ips" {
  type        = list(string)
  description = "IPs/CIDRs allowed to reach Postgres/Redis (e.g. your ACK node/pod CIDRs)"
  default     = ["10.0.0.0/8"]
}

variable "ack_cluster_name" {
  type        = string
  description = "Existing ACK cluster name regex to deploy the Helm chart into (optional reference)"
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

variable "db_instance_type" {
  type    = string
  default = "pg.n2.small.2c"
}

variable "postgres_version" {
  type    = string
  default = "16.0"
}

variable "redis_instance_class" {
  type    = string
  default = "redis.master.small.default"
}

variable "redis_version" {
  type    = string
  default = "7.0"
}

variable "redis_password" {
  type      = string
  sensitive = true
}

variable "tags" {
  type    = map(string)
  default = { app = "mixle-mlops" }
}
