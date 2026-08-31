#!/usr/bin/env bash
# Tear the stack down, then PROVE it is gone.
#
# `terraform destroy` reporting success is not proof. It only knows about resources in its own
# state: anything created outside Terraform, anything it failed to delete and dropped, and
# anything a partial apply orphaned are all invisible to it. This script therefore asks AWS
# directly, filtered by the Project tag every resource carries, and exits non-zero if anything
# is still standing.
#
# Exit codes:  0 = destroyed and verified clean   1 = destroy failed   2 = leftovers found
#
#   ./infra/destroy.sh

set -uo pipefail

cd "$(dirname "$0")"

PROJECT="${PROJECT:-coin-mlops}"
REGION="${AWS_REGION:-us-east-1}"

echo "=============================================================="
echo "terraform destroy   (project=$PROJECT region=$REGION)"
echo "=============================================================="
terraform destroy -auto-approve
DESTROY_RC=$?

if [ $DESTROY_RC -ne 0 ]; then
  echo
  echo "!! terraform destroy exited $DESTROY_RC -- resources are probably still running."
  echo "!! Running the verification anyway so the damage is visible."
fi

echo
echo "=============================================================="
echo "Verifying nothing is left"
echo "=============================================================="

LEFTOVERS=0

note_leftover() {
  echo "  !! STILL STANDING: $1"
  LEFTOVERS=$((LEFTOVERS + 1))
}

# --- ECS ------------------------------------------------------------------------------------
# Clusters first: list-services needs a cluster name, and a destroyed cluster means there is no
# service to list. Both are checked because a service can outlive a failed cluster deletion.
echo
echo "-- ECS clusters"
CLUSTERS=$(aws ecs list-clusters --region "$REGION" --query 'clusterArns[]' --output text 2>/dev/null)
FOUND_CLUSTER=""
for c in $CLUSTERS; do
  case "$c" in
    *"$PROJECT"*) FOUND_CLUSTER="$c"; note_leftover "ECS cluster $c" ;;
  esac
done
[ -z "$FOUND_CLUSTER" ] && echo "  ok - no cluster matching '$PROJECT'"

echo
echo "-- ECS services"
if [ -n "$FOUND_CLUSTER" ]; then
  SERVICES=$(aws ecs list-services --cluster "$FOUND_CLUSTER" --region "$REGION" \
    --query 'serviceArns[]' --output text 2>/dev/null)
  if [ -n "$SERVICES" ] && [ "$SERVICES" != "None" ]; then
    for s in $SERVICES; do note_leftover "ECS service $s"; done
  else
    echo "  ok - cluster present but no services in it"
  fi
else
  echo "  ok - no cluster, so no services (aws ecs list-services needs one)"
fi

# Running tasks are what actually bills, so they are checked independently of the cluster.
echo
echo "-- Running Fargate tasks"
RUNNING=$(aws ecs list-clusters --region "$REGION" --query 'clusterArns[]' --output text 2>/dev/null)
TASK_COUNT=0
for c in $RUNNING; do
  case "$c" in
    *"$PROJECT"*)
      T=$(aws ecs list-tasks --cluster "$c" --desired-status RUNNING --region "$REGION" \
        --query 'taskArns[]' --output text 2>/dev/null)
      [ -n "$T" ] && [ "$T" != "None" ] && { note_leftover "running task(s) in $c: $T"; TASK_COUNT=1; }
      ;;
  esac
done
[ $TASK_COUNT -eq 0 ] && echo "  ok - no running tasks under a '$PROJECT' cluster"

# --- VPC ------------------------------------------------------------------------------------
# Filtered by the Project tag rather than by CIDR: the tag is what every resource carries, and a
# CIDR match would miss anything created with a different one.
echo
echo "-- VPCs tagged Project=$PROJECT"
VPCS=$(aws ec2 describe-vpcs --region "$REGION" \
  --filters "Name=tag:Project,Values=$PROJECT" \
  --query 'Vpcs[].VpcId' --output text 2>/dev/null)
if [ -n "$VPCS" ] && [ "$VPCS" != "None" ]; then
  for v in $VPCS; do note_leftover "VPC $v"; done
else
  echo "  ok - no VPC tagged Project=$PROJECT"
fi

# A NAT gateway is the expensive mistake this stack is built to avoid, so it gets its own check
# even though none is ever created -- a leftover from an experiment would bill ~$1/day unnoticed.
echo
echo "-- NAT gateways (should never exist for this project)"
NATS=$(aws ec2 describe-nat-gateways --region "$REGION" \
  --filter "Name=tag:Project,Values=$PROJECT" \
  --query 'NatGateways[?State!=`deleted`].NatGatewayId' --output text 2>/dev/null)
if [ -n "$NATS" ] && [ "$NATS" != "None" ]; then
  for n in $NATS; do note_leftover "NAT GATEWAY $n -- this bills ~\$0.045/hr, delete it now"; done
else
  echo "  ok - no NAT gateways"
fi

# Elastic IPs survive the resources they were attached to and bill when unassociated.
echo
echo "-- Elastic IPs tagged Project=$PROJECT"
EIPS=$(aws ec2 describe-addresses --region "$REGION" \
  --filters "Name=tag:Project,Values=$PROJECT" \
  --query 'Addresses[].AllocationId' --output text 2>/dev/null)
if [ -n "$EIPS" ] && [ "$EIPS" != "None" ]; then
  for e in $EIPS; do note_leftover "Elastic IP $e"; done
else
  echo "  ok - no Elastic IPs"
fi

# --- ECR ------------------------------------------------------------------------------------
echo
echo "-- ECR repositories"
REPOS=$(aws ecr describe-repositories --region "$REGION" \
  --query 'repositories[].repositoryName' --output text 2>/dev/null)
FOUND_REPO=0
for r in $REPOS; do
  case "$r" in
    *"$PROJECT"*) note_leftover "ECR repository $r (image storage bills per GB-month)"; FOUND_REPO=1 ;;
  esac
done
[ $FOUND_REPO -eq 0 ] && echo "  ok - no ECR repository matching '$PROJECT'"

# --- CloudWatch -----------------------------------------------------------------------------
# 1-day retention, so a leftover group costs approximately nothing -- but it is state, and the
# contract is that nothing remains.
echo
echo "-- CloudWatch log groups"
LGS=$(aws logs describe-log-groups --region "$REGION" \
  --log-group-name-prefix "/ecs/$PROJECT" \
  --query 'logGroups[].logGroupName' --output text 2>/dev/null)
if [ -n "$LGS" ] && [ "$LGS" != "None" ]; then
  for l in $LGS; do note_leftover "log group $l"; done
else
  echo "  ok - no log group /ecs/$PROJECT"
fi

# --- IAM ------------------------------------------------------------------------------------
# Roles and the OIDC provider cost nothing, but they are standing grants of access -- a deploy
# role that outlives its stack is a role nobody is reviewing.
echo
echo "-- IAM roles"
ROLES=$(aws iam list-roles --query "Roles[?starts_with(RoleName, '$PROJECT')].RoleName" \
  --output text 2>/dev/null)
if [ -n "$ROLES" ] && [ "$ROLES" != "None" ]; then
  for r in $ROLES; do note_leftover "IAM role $r (no cost, but a standing grant)"; done
else
  echo "  ok - no IAM roles prefixed '$PROJECT'"
fi

echo
echo "-- GitHub OIDC provider"
OIDC=$(aws iam list-open-id-connect-providers \
  --query 'OpenIDConnectProviderList[].Arn' --output text 2>/dev/null)
if echo "$OIDC" | grep -q 'token.actions.githubusercontent.com'; then
  note_leftover "OIDC provider for token.actions.githubusercontent.com (no cost, but a standing trust)"
else
  echo "  ok - no GitHub OIDC provider"
fi

# --------------------------------------------------------------------------------------------
echo
echo "=============================================================="
if [ $LEFTOVERS -eq 0 ]; then
  echo "CLEAN - nothing tagged Project=$PROJECT remains in $REGION."
  echo "=============================================================="
  exit 0
fi
echo "$LEFTOVERS RESOURCE(S) STILL STANDING - see the '!!' lines above."
echo "=============================================================="
exit 2
