# Registry for the serve-cloud image. Nothing else is stored here.
resource "aws_ecr_repository" "app" {
  name = var.project

  # force_delete is load-bearing for the teardown contract. ECR refuses to delete a repository
  # that still holds images, so without this `terraform destroy` fails on the one resource that
  # keeps billing (storage), leaves the VPC half-destroyed, and needs a manual console visit.
  # A repository whose entire contents are one throwaway image has nothing worth protecting.
  force_delete = true

  image_scanning_configuration {
    scan_on_push = false # per-scan charge, and this image is not what is being demonstrated
  }

  tags = {
    Name = "${var.project}-ecr"
  }
}

# Storage is billed per GB-month beyond the 500 MB free tier and the cloud image is ~1.7 GB, so a
# repository left behind is a real (small) recurring charge. destroy.sh checks for it explicitly.
# This policy is belt-and-braces for the case where the repo outlives the stack anyway.
resource "aws_ecr_lifecycle_policy" "app" {
  repository = aws_ecr_repository.app.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep only the most recent image"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 1
      }
      action = { type = "expire" }
    }]
  })
}
