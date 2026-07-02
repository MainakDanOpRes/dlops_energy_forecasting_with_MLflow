# Deploying to AWS EC2 via GitHub Actions + ECR

This pipeline works like this:

```
push to main          → lint/test only (fast feedback, no deploy)
push a tag "vX.Y.Z"   → lint/test → build image → push to Amazon ECR (tagged vX.Y.Z + latest)
                            → self-hosted runner (on your EC2 box) pulls vX.Y.Z and restarts the container
```

Deploys only happen when you explicitly cut a release by pushing a version tag, e.g.:

```bash
git tag v1.0.0
git push origin v1.0.0
```

Ordinary commits to `main` just run lint/tests — nothing gets built, pushed, or deployed.

## 1. Create an ECR repository

```bash
aws ecr create-repository --repository-name energy-forecasting --region <your-region>
```

## 2. Create an IAM user for GitHub Actions

Create an IAM user (e.g. `github-actions-deployer`) with programmatic access and this policy
(scope the resource ARN down to your repo in production):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:PutImage",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload"
      ],
      "Resource": "*"
    }
  ]
}
```

Save the generated Access Key ID / Secret Access Key.

## 3. Launch the EC2 instance

- AMI: Ubuntu 22.04/24.04 LTS
- Instance type: `t3.large` or larger — you're running **two** copies of the model in memory
  (FastAPI's `PredictionPipeline()` + Streamlit's own `load_pipeline()`), on top of
  TensorFlow/PyTorch, so 4 GB RAM will be tight; 8 GB+ is safer.
- Security group: allow inbound **22** (SSH, your IP only), **8000** (FastAPI, `0.0.0.0/0`),
  and **8501** (Streamlit, `0.0.0.0/0`)
- Attach/create a key pair for SSH access

SSH in and install Docker + the Compose plugin:

```bash
sudo apt-get update -y
sudo apt-get install -y docker.io docker-compose-plugin
sudo systemctl enable docker --now
sudo usermod -aG docker ubuntu
# log out & back in so the group change takes effect
docker compose version   # sanity check
```

## 4. Register the EC2 instance as a GitHub Actions self-hosted runner

In your repo: **Settings → Actions → Runners → New self-hosted runner** (choose Linux/x64),
then run the commands GitHub shows you on the EC2 box, e.g.:

```bash
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64-<version>.tar.gz -L \
  https://github.com/actions/runner/releases/download/v<version>/actions-runner-linux-x64-<version>.tar.gz
tar xzf actions-runner-linux-x64-<version>.tar.gz
./config.sh --url https://github.com/<owner>/dlops_energy_forecasting_with_MLflow --token <token-from-github>

# Run it as a service so it survives reboots
sudo ./svc.sh install
sudo ./svc.sh start
```

The runner must be able to run `docker` — since it runs as the `ubuntu` user (in the `docker`
group from step 3), no `sudo` is needed inside workflow steps.

## 5. Add repository secrets

**Settings → Secrets and variables → Actions → New repository secret**:

| Secret | Value |
|---|---|
| `AWS_ACCESS_KEY_ID` | from step 2 |
| `AWS_SECRET_ACCESS_KEY` | from step 2 |
| `AWS_REGION` | e.g. `ap-south-1` |
| `ECR_REPOSITORY_NAME` | `energy-forecasting` |
| `MLFLOW_TRACKING_URI` | your MLflow tracking server URI (optional, if used at inference time) |

## 6. Cut a release

```bash
git tag v1.0.0
git push origin v1.0.0
```

This triggers the full pipeline. On success:

- FastAPI (Swagger docs) → `http://<EC2-public-IP>:8000/docs`
- Streamlit dashboard → `http://<EC2-public-IP>:8501`

both running the exact image tagged `v1.0.0` in ECR, deployed via `docker compose` from
`docker-compose.yml` at the repo root.

To roll back, just re-run the workflow for an older tag from **Actions → CI/CD → Run workflow**,
or re-push that tag's commit as a new tag (e.g. `v1.0.1-hotfix`).

## Notes

- `docker-compose.yml` runs `fastapi_app.py` (port 8000) and `streamlit_app.py` (port 8501) as
  two containers from the *same* image, each launched with a different command — no separate
  Dockerfile needed per app. `flask_app.py` isn't deployed by default; add a third service block
  in `docker-compose.yml` (same pattern, port `5000`) if you want it running too.
- Both containers load their own copy of the trained model into memory — expect roughly double
  the RAM of running just one service. If that's too heavy for your instance, consider having
  Streamlit call the FastAPI `/predict` and `/forecast` endpoints over HTTP instead of importing
  `PredictionPipeline` directly, so only one process holds the model.
- `tensorflow` + `torch` in `pyproject.toml` make for a large image; if you don't need both,
  drop the unused one to cut build time and image size significantly.
- For production, put both services behind an Nginx reverse proxy or an Application Load
  Balancer so you can expose everything on port 80/443 with TLS, instead of raw ports 8000/8501.
