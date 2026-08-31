# Public-subnet-only networking. There is NO NAT GATEWAY here, on purpose.
#
# A NAT gateway is ~$0.045/hr plus $0.045/GB processed -- it bills from the moment it is created,
# whether or not a task is running, and it is the single most likely way to overrun a small
# budget. It exists to give PRIVATE subnets outbound internet. This stack does not need it: the
# task runs in a public subnet with a public IP and an internet gateway route, which gives it
# both outbound (ECR pull, CloudWatch) and inbound (the curl that proves it serves) for $0.
#
# There is also NO LOAD BALANCER. An ALB is ~$16/month plus LCU charges and would add a stable
# DNS name, TLS termination and health-check-driven replacement -- none of which this is
# demonstrating. Traffic goes straight to the task's public IP.
#
# What that costs, stated plainly: the task's public IP changes every time ECS replaces the task,
# there is no TLS, and there is no redundancy. All three are correct trade-offs for a stack whose
# entire life is one apply, one curl, and one destroy -- and all three are wrong for anything
# that stays up.

data "aws_availability_zones" "available" {
  state = "available"
}

resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true # required for assign_public_ip to yield a resolvable task

  tags = {
    Name = "${var.project}-vpc"
  }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "${var.project}-igw"
  }
}

# Two subnets across two AZs. One task runs at a time, so the second subnet buys no redundancy
# today -- it is here because a Fargate service that can only place tasks in one AZ cannot be
# scaled or rescheduled when that AZ has no capacity, and adding a subnet later means recreating
# the service. Subnets are free.
resource "aws_subnet" "public" {
  count = 2

  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, count.index + 1)
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name = "${var.project}-public-${data.aws_availability_zones.available.names[count.index]}"
  }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  # The line that replaces the NAT gateway: 0.0.0.0/0 straight out the internet gateway.
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name = "${var.project}-public-rt"
  }
}

resource "aws_route_table_association" "public" {
  count = length(aws_subnet.public)

  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

# WIDE OPEN ON 8000, and that is a deliberate, time-boxed decision, not an oversight.
#
# /predict takes an unauthenticated image upload from anywhere. That is acceptable here for
# exactly one reason: this stack is created, curled, and destroyed inside a single session, so
# the exposure window is minutes and the thing exposed is a coin classifier with no data behind
# it. Anything that stays up needs the source CIDR narrowed to the caller, an ALB with TLS, and
# authentication in front of /predict.
resource "aws_security_group" "task" {
  name        = "${var.project}-task-sg"
  description = "Inbound ${var.container_port} from anywhere; all outbound."
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "FastAPI /health and /predict"
    from_port   = var.container_port
    to_port     = var.container_port
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Outbound is not optional: with no NAT, this is how the task pulls from ECR and writes to
  # CloudWatch Logs. Both go over the internet gateway to public AWS endpoints.
  egress {
    description = "ECR pull, CloudWatch Logs"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project}-task-sg"
  }
}
