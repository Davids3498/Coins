variable "aws_region" {
  description = "Region for every resource. us-east-1 matches the MLflow artifact bucket in the makefile."
  type        = string
  default     = "us-east-1"
}

variable "project" {
  description = "Value of the Project tag on every resource. infra/destroy.sh greps for it when verifying teardown."
  type        = string
  default     = "coin-mlops"
}

variable "image_tag" {
  description = "Tag of the serve-cloud image in ECR that the task definition runs."
  type        = string
  default     = "cloud"
}

variable "container_port" {
  description = "Port uvicorn binds in the container, and the only port the security group opens."
  type        = number
  default     = 8000
}

# 0.5 vCPU / 1 GB. The image carries CPU torch and unpickles a MobileNetV3-Large at startup, so
# the 0.25 vCPU / 512 MB tier is not worth attempting -- torch alone is ~800 MB resident. This is
# the second-smallest Fargate size and the cheapest one with headroom for the load.
variable "task_cpu" {
  description = "Fargate task CPU units (512 = 0.5 vCPU)."
  type        = number
  default     = 512
}

variable "task_memory" {
  description = "Fargate task memory in MiB."
  type        = number
  default     = 1024
}

variable "vpc_cidr" {
  description = "CIDR for the project VPC. Deliberately not 172.31/16 so it cannot collide with the account's default VPC."
  type        = string
  default     = "10.20.0.0/16"
}

variable "github_repo" {
  description = "owner/repo the CI deploy role's OIDC trust policy is scoped to. Nothing outside this repo can assume it."
  type        = string
  default     = "Davids3498/Coins"
}

variable "mlflow_artifact_bucket" {
  description = <<-EOT
    Bucket holding the MLflow artifact store. Read access is granted to the CI DEPLOY role only
    -- the runner needs it to export the champion before building the cloud image. The ECS task
    execution role gets no S3 at all: the model is baked into the image, so the running container
    never reaches S3.
  EOT
  type        = string
  default     = "davids-mlops-artifacts-8412"
}
