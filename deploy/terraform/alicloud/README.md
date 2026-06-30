# mixle-mlops on Alibaba Cloud (ACK + OSS + ApsaraDB RDS Postgres + ApsaraDB for Redis)

Provisions the managed dependencies and emits the env the Helm chart needs.

## Apply

```bash
cd deploy/terraform/alicloud
terraform init
terraform apply \
  -var bucket_name=my-mixle-objects \
  -var vswitch_id=vsw-xxxx \
  -var db_password='a-strong-password' \
  -var redis_password='another-strong-password' \
  -var ack_cluster_name=my-ack
```

## Then install the chart

```bash
helm upgrade --install mixle-mlops ../../helm/mixle-mlops \
  -f ../../helm/mixle-mlops/values-alicloud.yaml \
  --set objectStore.url="$(terraform output -raw object_store_url)" \
  --set objectStore.endpoint="$(terraform output -raw object_store_endpoint)" \
  --set database.url="$(terraform output -raw database_url)" \
  --set redis.url="$(terraform output -raw redis_url)"
```

## Credentials

Pods get OSS access via **ACK RRSA** (RAM Roles for Service Accounts) — annotate the chart ServiceAccount
(`serviceAccount.annotations.pod-identity.alibabacloud.com/role-name`) with a RAM role granting
`oss:GetObject/PutObject/DeleteObject/ListObjects` on the bucket. Set `MIXLE_OBJECT_STORE_ENDPOINT` to the OSS
region endpoint (the `ossfs` driver needs it).

The **same image and chart** run on every cloud — only the URLs change.

> terraform/docker are not installed in this dev sandbox; this is plain HCL.
