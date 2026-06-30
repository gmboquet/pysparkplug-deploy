# mixle-mlops on GCP (GKE + Cloud Storage + Cloud SQL + Memorystore)

Provisions the managed dependencies and emits the env the Helm chart needs.

## Apply

```bash
cd deploy/terraform/gcp
terraform init
terraform apply \
  -var project=my-project \
  -var bucket_name=my-mixle-objects \
  -var network=projects/my-project/global/networks/default \
  -var db_password='a-strong-password' \
  -var gke_cluster_name=my-gke
```

## Then install the chart

```bash
helm upgrade --install mixle-mlops ../../helm/mixle-mlops \
  -f ../../helm/mixle-mlops/values-gcp.yaml \
  --set objectStore.url="$(terraform output -raw object_store_url)" \
  --set database.url="$(terraform output -raw database_url)" \
  --set redis.url="$(terraform output -raw redis_url)"
```

## Credentials

Pods get GCS access via **Workload Identity** — bind the chart's KSA to a GSA with
`roles/storage.objectAdmin` on the bucket and annotate the ServiceAccount
(`serviceAccount.annotations.iam.gke.io/gcp-service-account`). No keys in the cluster.

The **same image and chart** run on every cloud — only the three URLs change.

> terraform/docker are not installed in this dev sandbox; this is plain HCL.
