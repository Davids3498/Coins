# Two identities, with deliberately different reach:
#
#   task_execution  What the ECS AGENT uses to start the task: pull the image, create the log
#                   stream. Nothing else. NO S3 -- the model is baked into the image, so the
#                   running container never reaches an artifact store.
#   ci_deploy       What a GitHub Actions run assumes via OIDC to build and ship. It needs more,
#                   and every extra permission is justified inline below.
#
# The application itself gets no role at all (see ecs.tf: no task_role_arn).

# ---------------------------------------------------------------------------------------------
# ECS task execution role
# ---------------------------------------------------------------------------------------------

data "aws_iam_policy_document" "ecs_tasks_assume" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "task_execution" {
  name               = "${var.project}-task-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume.json

  tags = {
    Name = "${var.project}-task-execution"
  }
}

# The AWS-managed policy, not a hand-rolled one: it is exactly ECR pull + CloudWatch Logs write,
# it is what every Fargate task needs, and AWS keeps it correct as the ECR API changes. Writing
# it by hand here would be a copy that drifts.
resource "aws_iam_role_policy_attachment" "task_execution" {
  role       = aws_iam_role.task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# ---------------------------------------------------------------------------------------------
# GitHub Actions OIDC -- CI to AWS with no long-lived keys
# ---------------------------------------------------------------------------------------------
#
# THE POINT: there is no AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY anywhere -- not in this repo,
# not in GitHub secrets, not on a developer's machine. A workflow run presents a short-lived
# token GitHub signed, AWS verifies it against GitHub's OIDC keys, and hands back credentials
# that expire with the job. A leaked access key is valid until someone notices; a leaked OIDC
# token is valid for minutes and only for this repository.
#
# It lives in Terraform rather than in the console for the same reason everything else here does:
# a trust policy nobody can read is a trust policy nobody can review. The condition below is the
# entire security boundary of this role, and it belongs in a diff.

resource "aws_iam_openid_connect_provider" "github" {
  url            = "https://token.actions.githubusercontent.com"
  client_id_list = ["sts.amazonaws.com"]

  # GitHub's OIDC certificate chain root. AWS stopped verifying this thumbprint for the GitHub
  # issuer in 2023 (it validates against the library of trusted CAs instead), but the API still
  # requires the field.
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]

  tags = {
    Name = "${var.project}-github-oidc"
  }
}

data "aws_iam_policy_document" "ci_deploy_assume" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    # THE SCOPE. Without this condition the role trusts every OIDC token GitHub issues -- which
    # is every repository on GitHub, including anyone's fork. "repo:owner/name:*" means any
    # branch, tag or environment OF THIS REPOSITORY and nothing else. Narrowing further
    # (":ref:refs/heads/master") is the next step if this ever deploys automatically; it is left
    # wide across refs because the deploy job is workflow_dispatch-only and gets run from
    # feature branches on purpose.
    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_repo}:*"]
    }
  }
}

resource "aws_iam_role" "ci_deploy" {
  name                 = "${var.project}-ci-deploy"
  assume_role_policy   = data.aws_iam_policy_document.ci_deploy_assume.json
  max_session_duration = 3600

  tags = {
    Name = "${var.project}-ci-deploy"
  }
}

data "aws_iam_policy_document" "ci_deploy" {
  # Push to ECR. GetAuthorizationToken cannot be resource-scoped -- the API takes no resource --
  # so it is a separate statement rather than silently widening the push statement to "*".
  statement {
    sid       = "EcrAuth"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    sid = "EcrPush"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:CompleteLayerUpload",
      "ecr:InitiateLayerUpload",
      "ecr:PutImage",
      "ecr:UploadLayerPart",
      "ecr:BatchGetImage",
      "ecr:GetDownloadUrlForLayer",
    ]
    resources = [aws_ecr_repository.app.arn] # this repository only
  }

  # Force a new deployment of the one service, and read enough to report what happened.
  # UpdateService is scoped to the service ARN; the Describe/List calls take no resource.
  statement {
    sid       = "EcsDeploy"
    actions   = ["ecs:UpdateService"]
    resources = [aws_ecs_service.app.id]
  }

  statement {
    sid = "EcsRead"
    actions = [
      "ecs:DescribeServices",
      "ecs:DescribeTasks",
      "ecs:ListTasks",
      "ecs:DescribeTaskDefinition",
    ]
    resources = ["*"]
  }

  # READ ON THE MLFLOW ARTIFACT BUCKET -- the one permission here that is not obviously required,
  # so: the deploy job builds serve-cloud.Dockerfile, which needs build/champion/, which
  # scripts/export_champion.py produces by downloading the champion from the registry. The
  # registry's artifact store IS this bucket. Without this the runner can resolve the alias to a
  # version and then fail to fetch a single byte of the model.
  #
  # Read-only and scoped to one bucket. The ECS task execution role gets NONE of this: the
  # running container has the model baked in and never calls S3.
  statement {
    sid     = "MlflowArtifactRead"
    actions = ["s3:GetObject", "s3:ListBucket"]
    resources = [
      "arn:aws:s3:::${var.mlflow_artifact_bucket}",
      "arn:aws:s3:::${var.mlflow_artifact_bucket}/*",
    ]
  }
}

resource "aws_iam_role_policy" "ci_deploy" {
  name   = "${var.project}-ci-deploy"
  role   = aws_iam_role.ci_deploy.id
  policy = data.aws_iam_policy_document.ci_deploy.json
}
