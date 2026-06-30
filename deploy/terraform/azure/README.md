# mixle-mlops on Azure (AKS + Blob Storage + PostgreSQL Flexible Server + Azure Cache for Redis)

Provisions the managed dependencies and emits the env the Helm chart needs.

## Apply

```bash
cd deploy/terraform/azure
terraform init
terraform apply \
  -var resource_group=mixle-rg \
  -var storage_account_name=mymixleobjects \
  -var db_password='a-strong-password' \
  -var aks_cluster_name=my-aks
```

## Then install the chart

```bash
helm upgrade --install mixle-mlops ../../helm/mixle-mlops \
  -f ../../helm/mixle-mlops/values-azure.yaml \
  --set objectStore.url="$(terraform output -raw object_store_url)" \
  --set objectStore.endpoint="$(terraform output -raw object_store_endpoint)" \
  --set database.url="$(terraform output -raw database_url)" \
  --set redis.url="$(terraform output -raw redis_url)"
```

## Credentials

Pods get Blob access via **AKS Workload Identity** — federate the chart's KSA to a managed identity with
`Storage Blob Data Contributor` on the account and set
`serviceAccount.annotations.azure.workload.identity/client-id` + the `azure.workload.identity/use: "true"` pod
label (see `values-azure.yaml`). The `adlfs` driver reads the account from `MIXLE_OBJECT_STORE_ENDPOINT`.

The **same image and chart** run on every cloud — only the URLs change.

> terraform/docker are not installed in this dev sandbox; this is plain HCL.
