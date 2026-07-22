#!/bin/bash
# inject-thunderagent.sh
# Extracts the DGDR-generated DGD, adds a ThunderAgent router component,
# adjusts the Frontend to round-robin (since ThunderAgent owns KV routing),
# and outputs the final DGD YAML.
#
# Usage:
#   ./inject-thunderagent.sh > thunderagent-dgd.yaml
#   kubectl apply -f thunderagent-dgd.yaml

set -euo pipefail

DGDR_NAME="${1:-thunderagent-demo}"
NAMESPACE="${2:-dynamo-system}"
MODEL_NAME="Qwen/Qwen3-0.6B"
IMAGE="docker.io/ashnam/dynamo-vllm-runtime:thunderagent"

echo "# Extracting generated DGD from DGDR '$DGDR_NAME'..." >&2

# Wait for DGDR to reach Ready
phase=""
while [[ "$phase" != "Ready" && "$phase" != "Failed" ]]; do
  phase=$(kubectl get dgdr "$DGDR_NAME" -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null || echo "Pending")
  echo "#   DGDR phase: $phase" >&2
  if [[ "$phase" == "Failed" ]]; then
    echo "ERROR: DGDR failed. Check: kubectl describe dgdr $DGDR_NAME -n $NAMESPACE" >&2
    exit 1
  fi
  if [[ "$phase" != "Ready" ]]; then
    sleep 10
  fi
done

# Extract the generated DGD
GENERATED=$(kubectl get dgdr "$DGDR_NAME" -n "$NAMESPACE" \
  -o jsonpath='{.status.profilingResults.selectedConfig}')

if [[ -z "$GENERATED" ]]; then
  echo "ERROR: No selectedConfig found in DGDR status" >&2
  exit 1
fi

echo "# Generated DGD extracted. Injecting ThunderAgent router..." >&2

# Use Python to parse the generated DGD JSON, add ThunderAgent, and output YAML
python3 -c "
import json, sys

generated = json.loads('''$GENERATED''')

# Ensure we have the services map
services = generated.get('spec', {}).get('services', {})

# Override the Frontend to use round-robin (ThunderAgent owns KV routing)
if 'Frontend' in services:
    fe = services['Frontend']
    eps = fe.get('extraPodSpec', {})
    mc = eps.get('mainContainer', {})
    # Ensure the image is our custom one
    mc['image'] = '$IMAGE'
    # Set env for round-robin mode and model cache
    existing_env = mc.get('env', [])
    env_names = {e['name'] for e in existing_env}
    if 'HF_HOME' not in env_names:
        existing_env.append({'name': 'HF_HOME', 'value': '/opt/models'})
    mc['env'] = existing_env
    # Update the command/args to use round-robin
    if 'args' in mc:
        args_str = ' '.join(mc['args']) if isinstance(mc['args'], list) else mc['args']
        args_str = args_str.replace('--router-mode kv', '--router-mode round-robin')
        mc['args'] = [args_str] if isinstance(mc['args'], list) and len(mc['args']) == 1 else mc['args']
    eps['mainContainer'] = mc
    fe['extraPodSpec'] = eps
    services['Frontend'] = fe

# Override VllmWorker image
for svc_name in list(services.keys()):
    if 'Worker' in svc_name or 'Vllm' in svc_name:
        svc = services[svc_name]
        eps = svc.get('extraPodSpec', {})
        mc = eps.get('mainContainer', {})
        mc['image'] = '$IMAGE'
        existing_env = mc.get('env', [])
        env_names = {e['name'] for e in existing_env}
        if 'HF_HOME' not in env_names:
            existing_env.append({'name': 'HF_HOME', 'value': '/opt/models'})
        mc['env'] = existing_env
        # Add KV events config to worker args
        if 'args' in mc:
            args_str = ' '.join(mc['args']) if isinstance(mc['args'], list) else mc['args']
            if 'kv-events-config' not in args_str:
                args_str += ' --kv-events-config \'\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:5571\",\"enable_kv_cache_events\":true}\''
            mc['args'] = [args_str] if isinstance(mc['args'], list) and len(mc['args']) == 1 else mc['args']
        eps['mainContainer'] = mc
        svc['extraPodSpec'] = eps
        services[svc_name] = svc

# Add the ThunderAgent router service
services['ThunderAgentRouter'] = {
    'componentType': 'router',
    'replicas': 1,
    'envFromSecret': 'hf-token-secret',
    'extraPodSpec': {
        'mainContainer': {
            'image': '$IMAGE',
            'command': ['/bin/sh', '-c'],
            'args': [
                'python3 -m dynamo.thunderagent_router '
                '--endpoint dynamo.backend.generate '
                '--model-name $MODEL_NAME '
                '--model-path $MODEL_NAME '
                '--router-block-size 16 '
                '--router-reset-states '
                '--pause-threshold 0.95 '
                '--pause-target 0.80 '
                '--soft-demote-threshold 0.80 '
                '--resume-hysteresis 0.10 '
                '--scheduler-interval-seconds 5.0'
            ],
            'env': [
                {'name': 'SERVED_MODEL_NAME', 'value': '$MODEL_NAME'},
                {'name': 'MODEL_PATH', 'value': '$MODEL_NAME'},
                {'name': 'HF_HOME', 'value': '/opt/models'},
            ],
        }
    },
    'volumeMounts': [{'name': 'model-cache', 'mountPoint': '/opt/models'}],
}

generated['spec']['services'] = services

# Set the name and ensure metadata
generated['metadata'] = generated.get('metadata', {})
generated['metadata']['name'] = 'thunderagent-demo'
generated['metadata']['namespace'] = '$NAMESPACE'
generated['apiVersion'] = 'nvidia.com/v1alpha1'
generated['kind'] = 'DynamoGraphDeployment'

# Output as YAML
import yaml
print(yaml.dump(generated, default_flow_style=False, sort_keys=False))
" 2>&1
