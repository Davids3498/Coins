# The task's public IP is assigned at task start, not at apply, so it cannot come from a
# resource attribute -- see infra/scripts/task_public_ip.sh. depends_on defers this read until
# after the service exists; the script then polls until a task is RUNNING with an ENI, which is
# also what makes `terraform apply` block until the service has stabilized.
#
# The honest caveat: this is a shell-out, not real Terraform state. The IP changes whenever ECS
# replaces the task, and this output only reflects the moment apply ran. Re-run
# `terraform refresh` (or the script directly) to get the current one.
data "external" "task_ip" {
  program = ["bash", "${path.module}/scripts/task_public_ip.sh"]

  query = {
    cluster = aws_ecs_cluster.main.name
    service = aws_ecs_service.app.name
    region  = var.aws_region
  }

  depends_on = [aws_ecs_service.app]
}

output "task_public_ip" {
  description = "Public IP of the running Fargate task. Empty if no task was RUNNING when apply finished."
  value       = data.external.task_ip.result.public_ip
}

output "task_ip_status" {
  description = "Why task_public_ip is empty, when it is."
  value       = data.external.task_ip.result.status
}

output "service_url" {
  description = "Base URL to curl. /health and /predict hang off this."
  value = (
    data.external.task_ip.result.public_ip == ""
    ? "unavailable -- ${data.external.task_ip.result.status}"
    : "http://${data.external.task_ip.result.public_ip}:${var.container_port}"
  )
}

output "ecr_repository_url" {
  description = "Push the serve-cloud image here."
  value       = aws_ecr_repository.app.repository_url
}

output "cluster_name" {
  description = "ECS cluster name. infra/destroy.sh and the CI deploy job both take it from here."
  value       = aws_ecs_cluster.main.name
}

output "service_name" {
  description = "ECS service name."
  value       = aws_ecs_service.app.name
}

output "ci_deploy_role_arn" {
  description = "Role the GitHub Actions deploy job assumes via OIDC. Set as the AWS_DEPLOY_ROLE_ARN repo variable."
  value       = aws_iam_role.ci_deploy.arn
}

output "log_group" {
  description = "CloudWatch log group carrying the task's stdout (1-day retention)."
  value       = aws_cloudwatch_log_group.app.name
}
