#!/usr/bin/env bash
# Resolve the running task's public IP, for the `task_public_ip` Terraform output.
#
# WHY A SCRIPT. A Fargate task's public IP is assigned to its ENI when the task starts, which is
# after `terraform apply` has created the service. No AWS provider resource or data source
# exposes it, so the only way to have it as a real output is to shell out. That is what the
# `external` data source in outputs.tf does.
#
# WHY IT NEVER EXITS NON-ZERO. A failing external data source fails the whole plan/apply -- and
# during `terraform destroy`, or on the first apply before a task has been placed, there
# legitimately is no IP. Erroring there would make the stack hard to tear down, which is the
# opposite of what this stack is for. So: no IP is reported as an empty string plus a status
# saying why, and the caller decides whether that is a problem.
#
# Reads a JSON object {cluster, service, region} on stdin; writes {public_ip, status} on stdout.
# Both are the external data source protocol -- flat objects of strings only.
set -uo pipefail

INPUT=$(cat)
CLUSTER=$(echo "$INPUT" | python3 -c 'import json,sys; print(json.load(sys.stdin)["cluster"])')
SERVICE=$(echo "$INPUT" | python3 -c 'import json,sys; print(json.load(sys.stdin)["service"])')
REGION=$(echo "$INPUT"  | python3 -c 'import json,sys; print(json.load(sys.stdin)["region"])')

emit() { printf '{"public_ip":"%s","status":"%s"}\n' "$1" "$2"; exit 0; }

# Bounded wait: a Fargate task pulling a ~1.7 GB image typically reaches RUNNING in 1-3 minutes.
DEADLINE=$(( $(date +%s) + 300 ))

while :; do
  TASK_ARN=$(aws ecs list-tasks --cluster "$CLUSTER" --service-name "$SERVICE" \
    --desired-status RUNNING --region "$REGION" \
    --query 'taskArns[0]' --output text 2>/dev/null)

  if [ -n "$TASK_ARN" ] && [ "$TASK_ARN" != "None" ]; then
    ENI=$(aws ecs describe-tasks --cluster "$CLUSTER" --tasks "$TASK_ARN" --region "$REGION" \
      --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value | [0]' \
      --output text 2>/dev/null)

    if [ -n "$ENI" ] && [ "$ENI" != "None" ]; then
      IP=$(aws ec2 describe-network-interfaces --network-interface-ids "$ENI" --region "$REGION" \
        --query 'NetworkInterfaces[0].Association.PublicIp' --output text 2>/dev/null)
      if [ -n "$IP" ] && [ "$IP" != "None" ]; then
        emit "$IP" "running"
      fi
    fi
  fi

  [ "$(date +%s)" -ge "$DEADLINE" ] && emit "" "no-running-task-within-300s"
  sleep 10
done
