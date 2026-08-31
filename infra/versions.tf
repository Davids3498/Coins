# Provider + backend wiring. Everything else in this directory is a resource file.
#
# STATE IS LOCAL AND GITIGNORED. The production answer is an S3 backend with a DynamoDB lock
# table, and this stack is small enough that the difference is invisible -- one operator, one
# machine, one apply at a time. It is omitted deliberately: a remote backend is itself two
# resources that outlive `terraform destroy` (the bucket holding the state cannot be destroyed
# by the state it holds), and this exercise's contract is that NOTHING is left standing. See the
# README's "Cloud deploy" section.

terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    external = {
      source  = "hashicorp/external"
      version = "~> 2.3"
    }
  }
}

provider "aws" {
  region = var.aws_region

  # The teardown check greps for this tag. Applied by the provider rather than resource by
  # resource so a newly added resource is findable by default -- a leftover you cannot find is
  # a leftover you keep paying for.
  default_tags {
    tags = {
      Project   = var.project
      ManagedBy = "terraform"
    }
  }
}
