# mixle-mlops multi-cloud setup (AWS · Azure · GCP · Alibaba Cloud · any S3-compatible)

**One image, one Helm chart, every cloud.** The platform is cloud-agnostic: switching providers changes only
three URLs — the **object store**, the **database**, and **redis**. The gateway container and the
`deploy/helm/mixle-mlops` chart are identical everywhere (EKS / AKS / GKE / ACK / any conformant Kubernetes).

```
                    ┌──────────────────────────────────────────────┐
  terraform apply   │  managed bucket + Postgres + Redis (per cloud)│
  (per provider) ─▶ │  outputs: object_store_url / database_url /   │
                    │           redis_url                           │
                    └───────────────────────┬──────────────────────┘
                                             │  (3 URLs)
                                             ▼
  helm upgrade --install  ─▶  the SAME mixle-mlops chart + image  ─▶  running gateway
```

## What selects the cloud

`MIXLE_OBJECT_STORE_URL` (see `mixle_mlops/storage/objectstore.py`):

| Cloud            | object store URL          | driver  | k8s          |
|------------------|---------------------------|---------|--------------|
| AWS / S3-compat  | `s3://bucket/prefix`      | `s3fs`  | EKS          |
| GCP              | `gs://bucket/prefix`      | `gcsfs` | GKE          |
| Azure            | `az://container/prefix`   | `adlfs` | AKS          |
| Alibaba Cloud    | `oss://bucket/prefix`     | `ossfs` | ACK          |
| Local / dev      | `file://./mixle_data/objects` | builtin | any / laptop |

The database is a `postgresql+psycopg://…` URL (`MIXLE_DATABASE_URL`); redis is `redis://…`/`rediss://…`
(`MIXLE_REDIS_URL`). Drop any of them and the platform falls back to its local default (SQLite / local fs).

## One-command-ish quickstart (any provider)

```bash
# 0) scaffold a .env for your cloud (writes the right MIXLE_* keys with placeholders)
mixle-mlops init-cloud aws        # or: azure | gcp | alicloud | local

# 1) provision managed bucket + Postgres + Redis
cd deploy/terraform/aws           # or azure | gcp | alicloud
terraform init && terraform apply # see the per-provider README for required -var flags

# 2) install the SAME chart, fed the terraform outputs
helm upgrade --install mixle-mlops ../../helm/mixle-mlops \
  -f ../../helm/mixle-mlops/values-aws.yaml \
  --set objectStore.url="$(terraform output -raw object_store_url)" \
  --set database.url="$(terraform output -raw database_url)" \
  --set redis.url="$(terraform output -raw redis_url)"
```

Per-provider details (required variables, identity wiring): see
`deploy/terraform/{aws,azure,gcp,alicloud}/README.md`.

## Credentials: no static keys in the cluster

Each provider's chart values wire **workload identity** so the object-store driver authenticates without keys:

- AWS **IRSA** → `serviceAccount.annotations.eks.amazonaws.com/role-arn`
- GCP **Workload Identity** → `serviceAccount.annotations.iam.gke.io/gcp-service-account`
- Azure **Workload Identity** → `serviceAccount.annotations.azure.workload.identity/client-id`
- Alibaba **ACK RRSA** → `serviceAccount.annotations.pod-identity.alibabacloud.com/role-name`

## S3-compatible (MinIO / R2 / Wasabi / on-prem)

Use `s3://bucket` plus `objectStore.endpoint=https://your-endpoint`. No code or chart change.

## Local / laptop

`mixle-mlops init-cloud local` (or just run `mixle-serve`) uses SQLite + the local filesystem object store — the
exact same image, no cloud at all.

> Note: `terraform`, `helm`, and `docker` are not installed in the dev sandbox these files were authored in.
> The Terraform is plain HCL and the chart is standard Helm; both run unchanged where those tools are present.
