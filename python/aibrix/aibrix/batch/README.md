# AIBrix Batch Processing

This directory contains the batch processing system for AIBrix, designed to handle large-scale batch inference jobs using a sidecar pattern with vLLM.

## Asynchronous video batches

Create a Batch with `endpoint: "/v1/videos"` and POST rows in the usual JSONL
envelope. The first supported wire protocol is vLLM-Omni's Videos API:

```jsonl
{"custom_id":"video-001","method":"POST","url":"/v1/videos","body":{"model":"minimax-h3","prompt":"Continue the motion shown in the reference clip.","video_reference":{"video_url":"https://media.example.com/reference.mp4"},"num_inference_steps":50,"flow_shift":12.0,"extra_params":{"task":"ref2va","duration":4.0,"audio_flow_shift":3.0}}}
```

Replace the example URL with a backend-accessible reference. Images use
`image_reference.image_url`. Reference objects/lists and `extra_params` are
vLLM-Omni extensions and must match the deployed engine version. Scalars become
text form fields; objects and lists become JSON form fields in a multipart
request. Null fields are omitted. Uploaded file IDs and worker-local paths are
not resolved into binary uploads. Existing chat `messages` are not converted:
the dataset must explicitly provide the Videos prompt and media fields.

The execution pipeline is:

```text
MDS input validation
  -> driver reads and locks a row
  -> DispatchEngine admits the row and selects a channel
  -> POST /v1/videos and save its ID/destination/deadline
  -> GET /v1/videos/{id} until completed or failed
  -> stream GET /v1/videos/{id}/content to a temporary file
  -> persist the video to Files storage
  -> write result/error JSONL and update request counts
  -> release the concurrency slot
```

A successful result preserves the final Video metadata and adds
`response.body.output_file_id`. Download that file with
`GET /v1/files/{output_file_id}/content`. The Batch object's top-level
`output_file_id` still identifies the result JSONL. Video files use deterministic
IDs, remain available after the engine exits, and follow existing Files deletion
behavior. Temporary downloads are removed after persistence.

Use `extra_body.aibrix.client.max_concurrency` to bound active video rows and
`extra_body.aibrix.client.request_timeout_seconds` for their total execution
budget (creation, polling, download, and video persistence). The default is
`INFERENCE_TASK_TIMEOUT`; the Batch expiration also caps this budget. Polling
does not extend it. Result writing has a separate 30-second bound; the processing
lock reserves an additional minute for bounded cancellation and result cleanup.
Explicit cancellation or timeout attempts a DELETE for at most five seconds.
Ordinary worker interruption preserves the execution handle for resumption.

Status/content retries use `client.retry_policy` against the saved destination.
Creation retries are allowed for connection/pool failures before submission and
HTTP 429. An ambiguous creation timeout or server error never triggers automatic
POST replay. A submission marker is saved before POST: recovery without a saved
ID reports an unknown outcome, including a crash between that marker and POST.
This is conservative recovery, not an exactly-once submission guarantee.
After the ID is saved, recovery resumes the same job. A saved final response
allows result-write retries without downloading or generating again.

Gateway connections reuse the gateway's public video ID and owning-Pod routing.
Direct connections must identify the original backend (for example, Pod
endpoints from EndpointSlice discovery); a multi-replica ClusterIP cannot
guarantee this. The worker does not regenerate jobs whose owner or ID is gone.

Video rows use fixed admission limits even if LLM adaptive concurrency is
enabled. Polling yields to other admitted rows every two seconds. The existing
injectable `ConcurrencyController` remains the extension point for future
video-specific admission based on queueing, latency, throughput, and GPU
measurements. HTTP concurrency alone does not imply improved SMA utilization;
backend overlap requires measurement. Dry-run only echoes request construction.

Design: [RFC #3247](https://github.com/vllm-project/aibrix/issues/3247).

## Architecture Overview

The batch processing system consists of several key components:

- **BatchDriver**: Main orchestrator that manages job lifecycle
- **JobManager**: Handles job state management and tracking  
- **JobDriver**: Bridge between job management and inference execution
- **BatchWorker**: Script-based worker that runs in Kubernetes jobs

## Worker Architecture

The `worker.py` script implements a **sidecar pattern** where:

1. **batch-worker container**: Runs the worker script (exits when complete)
2. **llm-engine container**: Runs vLLM as a persistent service

### Workflow

1. Both containers start simultaneously in the same pod
2. Worker waits for vLLM health check on `localhost:8000`
3. Worker loads its parent Kubernetes Job spec using the Kubernetes API
4. Worker transforms the K8s Job to BatchJob using BatchJobTransformer
5. Worker creates BatchDriver and executes the job
6. Worker tracks job until FINALIZING state
7. Worker exits with status code 0
8. Kubernetes sends SIGTERM to vLLM container
9. Pod completes successfully

## Testing in Kubernetes

### Prerequisites

1. **Docker Images**: Ensure you have the required images built:
   ```bash
   # Build the AIBrix runtime image with batch worker
   make docker-build-runtime
   
   # Or use development images
   docker build -t aibrix/runtime:nightly .
   docker build -t aibrix/vllm-mock:nightly ./vllm-mock
   ```

2. **Kubernetes Cluster**: Have a running cluster (minikube, kind, or full cluster)

3. **Storage Setup**: Configure storage backend (S3, TOS, or local storage)

### Basic Testing

#### 1. Prepare Test Input File and Job Annotations

Create a JSONL file with batch requests and prepare job annotations:

```bash
# Create test input file
cat > /tmp/batch_input.jsonl << 'EOF'
{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hello world"}]}}
{"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "What is AI?"}]}}
EOF

# Upload to your storage backend and get the file ID
INPUT_FILE_ID="your-uploaded-file-id"
```

#### 2. Create and Run Batch Job

```bash
# Copy the job template
cp aibrix/metadata/setting/k8s_job_template.yaml /tmp/test-batch-job.yaml

# Edit the template with your specific values
JOB_NAME="test-batch-job-$(date +%s)"
sed -i "s/name: batch-job-template/name: ${JOB_NAME}/" /tmp/test-batch-job.yaml

# Add batch job annotations with your job specification
cat >> /tmp/test-batch-job.yaml << EOF
  annotations:
    batch.job.aibrix.ai/input-file-id: "${INPUT_FILE_ID}"
    batch.job.aibrix.ai/endpoint: "/v1/chat/completions"
    batch.job.aibrix.ai/completion-window: "24h"
    batch.job.aibrix.ai/session-id: "test-session-$(date +%s)"
EOF

# Apply the job
kubectl apply -f /tmp/test-batch-job.yaml
```

#### 3. Monitor Job Progress

```bash
# Watch job status
kubectl get jobs -w

# Check pod logs
POD_NAME=$(kubectl get pods -l app=aibrix-batch --no-headers -o custom-columns=":metadata.name" | head -1)

# Watch worker logs
kubectl logs -f $POD_NAME -c batch-worker

# Watch vLLM logs
kubectl logs -f $POD_NAME -c llm-engine
```

#### 4. Verify Job Completion

```bash
# Check job completion
kubectl get job test-batch-job-* -o wide

# Expected output should show:
# - COMPLETIONS: 1/1
# - STATUS: Complete

# Check pod status
kubectl get pods -l app=aibrix-batch

# Expected output should show:
# - STATUS: Completed
```

### Advanced Testing

#### Job Annotation Configuration

The worker loads job specifications from Kubernetes Job annotations using the `batch.job.aibrix.ai/` prefix:

```yaml
metadata:
  annotations:
    # Required annotations
    batch.job.aibrix.ai/input-file-id: "file-abc123"        # Input file with batch requests
    batch.job.aibrix.ai/endpoint: "/v1/chat/completions"    # API endpoint
    
    # Optional annotations  
    batch.job.aibrix.ai/completion-window: "24h"            # Job completion window (default: 24h)
    batch.job.aibrix.ai/session-id: "session-123"          # Session identifier
    batch.job.aibrix.ai/metadata.key1: "value1"            # Custom metadata (up to 16 pairs)
    batch.job.aibrix.ai/metadata.key2: "value2"

# Environment variables for worker configuration
spec:
  template:
    spec:
      containers:
      - name: batch-worker
        env:
        - name: STORAGE_TYPE
          value: "S3"                 # Optional: Storage backend (AUTO, LOCAL, S3, TOS)
        - name: METASTORE_TYPE  
          value: "S3"                 # Optional: Metastore backend (AUTO, LOCAL, S3, TOS)
```

#### Storage Backend Testing

For **S3 Backend**:
```yaml
env:
- name: STORAGE_TYPE
  value: "S3"
- name: AWS_ACCESS_KEY_ID
  value: "your-access-key"
- name: AWS_SECRET_ACCESS_KEY
  value: "your-secret-key"  
- name: AWS_REGION
  value: "us-west-2"
- name: S3_BUCKET
  value: "your-batch-bucket"
```

For **TOS Backend**:
```yaml
env:
- name: STORAGE_TYPE
  value: "TOS"
- name: TOS_ACCESS_KEY
  value: "your-tos-key"
- name: TOS_SECRET_KEY
  value: "your-tos-secret"
- name: TOS_ENDPOINT
  value: "https://tos-s3-cn-beijing.volces.com"
- name: TOS_REGION
  value: "cn-beijing"
```

#### Custom vLLM Configuration

Modify the vLLM container configuration:

```yaml
- name: llm-engine
  image: aibrix/vllm:nightly
  args:
    - --model
    - microsoft/DialoGPT-medium
    - --port
    - "8000"
    - --served-model-name
    - gpt-3.5-turbo
  resources:
    requests:
      nvidia.com/gpu: 1
    limits:
      nvidia.com/gpu: 1
```

### Debugging

#### Common Issues

1. **Worker fails to start**: Check job annotations and environment variables
   ```bash
   kubectl describe pod $POD_NAME
   kubectl logs $POD_NAME -c batch-worker
   
   # Check job annotations
   kubectl get job $JOB_NAME -o yaml | grep -A 10 annotations
   ```

2. **vLLM health check fails**: Check vLLM container logs
   ```bash
   kubectl logs $POD_NAME -c llm-engine
   ```

3. **Storage access issues**: Verify credentials and permissions
   ```bash
   kubectl logs $POD_NAME -c batch-worker | grep -i storage
   ```

4. **Job hangs**: Check both container logs and job events
   ```bash
   kubectl describe job test-batch-job-*
   kubectl get events --sort-by=.metadata.creationTimestamp
   ```

#### Debug Mode

Enable debug logging by adding:
```yaml
env:
- name: LOG_LEVEL
  value: "DEBUG"
```

### Cleanup

```bash
# Delete test job and pods
kubectl delete job test-batch-job-*
kubectl delete pod -l app=aibrix-batch

# Clean up test files
rm -f /tmp/test-batch-job.yaml /tmp/batch_input.jsonl
```

## Development

### Running Tests

```bash
# Install dependencies
poetry install --no-root --with dev --extras gpu-optimizer

# Run batch-specific tests
pytest tests/ -k batch

# Run formatting and linting
bash ./scripts/format.sh
```

### Code Structure

```
aibrix/batch/
├── README.md              # This file
├── worker.py              # Main worker script  
├── driver.py              # BatchDriver orchestrator
├── job_manager.py         # Job state management
├── job_driver.py          # Execution bridge
├── scheduler.py           # Job scheduling
├── job_entity/            # Job models and states
│   ├── batch_job.py       # BatchJob schema
│   └── k8s_transformer.py # Kubernetes integration
└── storage/               # Storage abstractions
    ├── adapter.py         # Storage bridge
    └── batch_storage.py   # Batch storage interface
```

The worker script follows the sidecar pattern and integrates with the existing batch processing architecture to provide scalable, Kubernetes-native batch inference capabilities.

## Architecture Changes

### New Kubernetes Integration

The worker now uses **Kubernetes-native job specification** instead of environment variables:

1. **Job Annotations**: Job specifications are stored as Kubernetes annotations using the `batch.job.aibrix.ai/` prefix
2. **Kubernetes API**: Worker connects to the K8s API server using in-cluster configuration 
3. **BatchJobTransformer**: Automatically transforms K8s Job objects to internal BatchJob models
4. **Service Account**: Requires appropriate RBAC permissions to read Job objects

### Required RBAC

The batch worker requires a service account with permissions to read Job objects, as defined in the aibrix/metadata/setting/k8s_job_rbac.yaml

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: job-reader-sa
  namespace: default
---
# Service Account for job pod
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: job-reader-role
  namespace: default
rules:
- apiGroups: ["batch"]
  resources: ["jobs"]
  verbs: ["get"] # Get permissions only
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: job-reader-binding
  namespace: default
subjects:
- kind: ServiceAccount
  name: job-reader-sa
  namespace: default
roleRef:
  kind: Role
  name: job-reader-role
  apiGroup: rbac.authorization.k8s.io
```

Add the service account to your job template:

```yaml
spec:
  template:
    spec:
      serviceAccountName: job-reader-sa
```
