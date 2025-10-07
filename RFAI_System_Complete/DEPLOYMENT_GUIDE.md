# RFAI Production Deployment Guide

This guide packages the Recursive Fractal Autonomous Intelligence (RFAI) system as a FastAPI service and provides the manifest files required to operate it on a Kubernetes cluster with automated TLS via cert-manager and Let's Encrypt.

## 1. Build and Publish the Container Image

1. Ensure Docker is installed and authenticated against the registry that will host the image.
2. From the `RFAI_System_Complete` directory, build the image:

   ```bash
   docker build -t <your-registry>/rfai-api:latest .
   ```

3. Push the image so it is accessible to your cluster:

   ```bash
   docker push <your-registry>/rfai-api:latest
   ```

Replace `<your-registry>` with your container registry (for example, `ghcr.io/your-org`).

## 2. Prepare Kubernetes Prerequisites

Install the NGINX Ingress controller and cert-manager (only once per cluster):

```bash
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.0/deploy/static/provider/cloud/deploy.yaml
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/latest/download/cert-manager.yaml
```

> **Tip:** Wait until all pods in the `ingress-nginx` and `cert-manager` namespaces report a `Running` status before continuing.

## 3. Configure ACME Issuer

Apply the ClusterIssuer manifest to enable certificate automation with Let's Encrypt:

```bash
kubectl apply -f deploy/clusterissuer.yaml
```

Update the email address in the manifest if required—it is used for expiry notifications from Let's Encrypt.

## 4. Deploy the RFAI Workload

1. Edit `deploy/rfai-app.yaml` and set the container image to the registry location used in step 1.
2. Apply the deployment and service manifests:

   ```bash
   kubectl apply -f deploy/rfai-app.yaml
   ```

3. Confirm the pods are running and healthy:

   ```bash
   kubectl get pods -l app=rfai-prometheus
   ```

The deployment includes liveness and readiness probes that query the `/health` endpoint of the FastAPI service.

## 5. Expose the Service Securely

1. Update `deploy/rfai-ingress.yaml` with the DNS name you will use to reach the API (replace `rfai.your-domain.com`).
2. Apply the ingress manifest:

   ```bash
   kubectl apply -f deploy/rfai-ingress.yaml
   ```

3. Point your public DNS record at the external IP address of the NGINX Ingress controller. Once DNS propagates, cert-manager will request and provision a TLS certificate automatically.

## 6. Runtime Verification

After the ingress is ready, validate the deployment:

```bash
curl https://rfai.your-domain.com/health
curl -X POST https://rfai.your-domain.com/process_task \
  -H "Content-Type: application/json" \
  -d '{"id": "demo", "type": "analysis", "complexity": 0.42, "data": [0.1, 0.2, 0.3]}'
```

You should receive JSON responses confirming the system status and processed task result. The API automatically serialises numpy-based outputs for compatibility with typical HTTP clients.

## 7. Ongoing Operations

- **Scaling:** Adjust `spec.replicas` in `deploy/rfai-app.yaml` to modify capacity. Horizontal Pod Autoscalers can also be layered on.
- **Observability:** Forward application logs via your preferred logging stack or integrate with Prometheus/Grafana for metrics.
- **Updates:** Rebuild and push a new image, update the deployment manifest with the new tag, and reapply it. Kubernetes will orchestrate a rolling update.
- **Backups:** Persist any generated artefacts or stateful data outside the container (for example, object storage or a database) because pods are ephemeral.

Following these steps will provide a highly-available, TLS-secured RFAI deployment backed by Kubernetes best practices.
