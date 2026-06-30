# Outputs map 1:1 onto the Helm chart's object store / database / redis knobs.

output "object_store_url" {
  value       = "s3://${aws_s3_bucket.objects.bucket}"
  description = "Set as MIXLE_OBJECT_STORE_URL / Helm objectStore.url"
}

output "database_url" {
  value       = "postgresql+psycopg://${var.db_username}:${var.db_password}@${aws_db_instance.pg.address}:5432/mixle"
  description = "Set as MIXLE_DATABASE_URL / Helm database.url"
  sensitive   = true
}

output "redis_url" {
  value       = "redis://${aws_elasticache_cluster.redis.cache_nodes[0].address}:6379/0"
  description = "Set as MIXLE_REDIS_URL / Helm redis.url"
}

output "region" {
  value = var.region
}
