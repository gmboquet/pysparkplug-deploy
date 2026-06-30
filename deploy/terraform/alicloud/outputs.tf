output "object_store_url" {
  value       = "oss://${alicloud_oss_bucket.objects.bucket}"
  description = "Set as MIXLE_OBJECT_STORE_URL / Helm objectStore.url"
}

output "object_store_endpoint" {
  value       = "https://oss-${var.region}.aliyuncs.com"
  description = "Set as MIXLE_OBJECT_STORE_ENDPOINT / Helm objectStore.endpoint (OSS region endpoint)"
}

output "database_url" {
  value       = "postgresql+psycopg://${var.db_username}:${var.db_password}@${alicloud_db_instance.pg.connection_string}:5432/mixle"
  description = "Set as MIXLE_DATABASE_URL / Helm database.url"
  sensitive   = true
}

output "redis_url" {
  value       = "redis://:${var.redis_password}@${alicloud_kvstore_instance.redis.connection_domain}:6379/0"
  description = "Set as MIXLE_REDIS_URL / Helm redis.url"
  sensitive   = true
}
