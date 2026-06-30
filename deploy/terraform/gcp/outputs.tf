output "object_store_url" {
  value       = "gs://${google_storage_bucket.objects.name}"
  description = "Set as MIXLE_OBJECT_STORE_URL / Helm objectStore.url"
}

output "database_url" {
  value       = "postgresql+psycopg://${var.db_username}:${var.db_password}@${google_sql_database_instance.pg.private_ip_address}:5432/mixle"
  description = "Set as MIXLE_DATABASE_URL / Helm database.url"
  sensitive   = true
}

output "redis_url" {
  value       = "redis://${google_redis_instance.redis.host}:${google_redis_instance.redis.port}/0"
  description = "Set as MIXLE_REDIS_URL / Helm redis.url"
}
