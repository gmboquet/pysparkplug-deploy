# mixle-mlops on AWS (EKS + S3 + RDS + ElastiCache)

Provisions the managed dependencies and emits the env the Helm chart needs. Assumes you already have a VPC,
private subnets, and (optionally) an EKS cluster.

## Apply

```bash
cd deploy/terraform/aws
terraform init
terraform apply \
  -var bucket_name=my-mixle-objects \
  -var vpc_id=vpc-xxxx \
  -var 'subnet_ids=["subnet-a","subnet-b"]' \
  -var db_password='a-strong-password' \
  -var eks_cluster_name=my-eks
```

## Then install the chart

```bash
terraform output -raw object_store_url   # s3://my-mixle-objects
terraform output -raw database_url
terraform output -raw redis_url

helm upgrade --install mixle-mlops ../../helm/mixle-mlops \
  -f ../../helm/mixle-mlops/values-aws.yaml \
  --set objectStore.url="$(terraform output -raw object_store_url)" \
  --set database.url="$(terraform output -raw database_url)" \
  --set redis.url="$(terraform output -raw redis_url)"
```

## Credentials

The gateway pods get S3 access via **IRSA** — annotate the chart ServiceAccount with the IAM role ARN
(`serviceAccount.annotations.eks.amazonaws.com/role-arn`). No static keys live in the cluster. The bucket policy
needed is `s3:GetObject/PutObject/DeleteObject/ListBucket` on the bucket.

The **same image and chart** run on every cloud — only `object_store_url` / `database_url` / `redis_url` change.

> terraform and docker are not installed in this dev sandbox; the module is plain HCL and applies in any
> environment with the AWS provider configured.
