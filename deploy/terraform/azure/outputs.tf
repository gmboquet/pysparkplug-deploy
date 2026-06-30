output "object_store_url" {
  value       = "az://${azurerm_storage_container.objects.name}"
  description = "Set as MIXLE_OBJECT_STORE_URL / Helm objectStore.url"
}

output "object_store_endpoint" {
  value       = azurerm_storage_account.objects.primary_blob_endpoint
  description = "Set as MIXLE_OBJECT_STORE_ENDPOINT / Helm objectStore.endpoint (the Blob account URL)"
}

output "database_url" {
  value       = "postgresql+psycopg://${var.db_username}:${var.db_password}@${azurerm_postgresql_flexible_server.pg.fqdn}:5432/mixle"
  description = "Set as MIXLE_DATABASE_URL / Helm database.url"
  sensitive   = true
}

output "redis_url" {
  value       = "rediss://${azurerm_redis_cache.redis.hostname}:${azurerm_redis_cache.redis.ssl_port}/0"
  description = "Set as MIXLE_REDIS_URL / Helm redis.url (TLS)"
  sensitive   = true
}
