# One Fargate service, one task, no autoscaling and no load balancer.

# retention_in_days = 1 because these logs exist to answer "did the task start and serve?" for
# the length of one session. CloudWatch bills ingestion and storage; a group created with the
# default "never expire" retention is a small permanent charge for logs nobody will read again.
resource "aws_cloudwatch_log_group" "app" {
  name              = "/ecs/${var.project}"
  retention_in_days = 1

  tags = {
    Name = "${var.project}-logs"
  }
}

# No container insights: it is an additional CloudWatch metrics charge, and with one task there
# is nothing to aggregate.
resource "aws_ecs_cluster" "main" {
  name = "${var.project}-cluster"

  setting {
    name  = "containerInsights"
    value = "disabled"
  }

  tags = {
    Name = "${var.project}-cluster"
  }
}

resource "aws_ecs_task_definition" "app" {
  family                   = var.project
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc" # the only mode Fargate supports; gives the task its own ENI
  cpu                      = var.task_cpu
  memory                   = var.task_memory

  # Execution role only. There is deliberately NO task_role_arn: the application makes no AWS API
  # calls at all -- the model is baked into the image, so it needs neither S3 nor the MLflow
  # tracking server. A task with no role cannot be used to reach anything in the account.
  execution_role_arn = aws_iam_role.task_execution.arn

  # The image is built on x86_64. Stating it explicitly means a build from an arm64 machine fails
  # at deploy with a platform mismatch rather than starting and crash-looping on an exec format
  # error, which reads as an application bug.
  runtime_platform {
    cpu_architecture        = "X86_64"
    operating_system_family = "LINUX"
  }

  container_definitions = jsonencode([{
    name      = var.project
    image     = "${aws_ecr_repository.app.repository_url}:${var.image_tag}"
    essential = true

    portMappings = [{
      containerPort = var.container_port
      protocol      = "tcp"
    }]

    # MODEL_SOURCE is already baked into serve-cloud.Dockerfile. It is repeated here so the task
    # definition alone answers "which load path does this run?" without reading a Dockerfile --
    # and so a base-image tag mixup cannot silently start the registry path with no tracking
    # server, which would fail at startup with a connection error instead of an honest one.
    environment = [
      { name = "MODEL_SOURCE", value = "local" },
      { name = "BAKED_MODEL_DIR", value = "/srv/model" },
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.app.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "ecs"
      }
    }
  }])

  tags = {
    Name = "${var.project}-taskdef"
  }
}

resource "aws_ecs_service" "app" {
  name            = "${var.project}-svc"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  # 0/100 rather than the 100/200 default. A force-new-deployment (what the CI deploy job does)
  # would otherwise start a second task before draining the first, doubling the Fargate bill for
  # the overlap. With one task and no load balancer there is nothing to keep available, so
  # replacing in place is both cheaper and simpler.
  deployment_minimum_healthy_percent = 0
  deployment_maximum_percent         = 100

  network_configuration {
    subnets         = aws_subnet.public[*].id
    security_groups = [aws_security_group.task.id]

    # The NAT-gateway replacement, stated at the one place it takes effect. Without a public IP
    # the task cannot reach ECR to pull its own image and never starts, and the only other way
    # to give it egress is a NAT gateway or VPC endpoints.
    assign_public_ip = true
  }

  # Without this the service can be created before the role's policy attachment lands, and the
  # first task fails to pull with an authorization error that looks like a bad image.
  depends_on = [aws_iam_role_policy_attachment.task_execution]

  tags = {
    Name = "${var.project}-svc"
  }
}
