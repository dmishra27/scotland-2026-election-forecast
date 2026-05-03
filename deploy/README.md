# Oracle Cloud Free Tier Deployment Guide

Scotland 2026 Election Forecast — permanent 24/7 hosting at £0/month.

## Architecture

```
Internet → OCI VCN (port 80/443) → nginx (Docker)
                                    ├─ /            → Streamlit  :8501
                                    ├─ /api/        → FastAPI    :8000
                                    └─ /mlflow/     → MLflow     :5000
```

All four services run in Docker Compose on a single
**VM.Standard.A1.Flex** (4 OCPU, 24 GB RAM, ARM64 Ampere) —
Oracle's permanently-free shape.

---

## 1. Create an Oracle Cloud account

1. Go to **cloud.oracle.com → Start for free**
2. Enter name, email, home region (pick closest for latency)
3. Verify email → add credit card (not charged; required for identity)
4. Account activates within 5 minutes

> The Always Free tier includes up to **4 ARM OCPUs + 24 GB RAM**
> distributed across Ampere A1 instances. One instance using all four
> OCPUs and 24 GB is permanently free.

---

## 2. Provision the ARM VM

### 2a. Create a Compute Instance

1. OCI Console → **Compute → Instances → Create instance**
2. **Name**: `scotland-forecast`
3. **Image**: Ubuntu 22.04 (click *Change image* → Canonical Ubuntu 22.04)
4. **Shape**: click *Change shape*
   - Shape series: **Ampere**
   - Shape name: `VM.Standard.A1.Flex`
   - OCPUs: **4** | Memory: **24 GB**
5. **SSH keys**: paste your `~/.ssh/id_rsa.pub` (or generate a new key pair
   and download the private key — you'll need it to SSH in)
6. Leave all other settings at defaults → **Create**

The instance boots in ~2 minutes. Note the **Public IP address**.

### 2b. Open VCN Security Rules

The VM's firewall has two layers: OCI Security Rules (cloud level) and
iptables (OS level). `setup_server.sh` handles iptables. You must open
the OCI rules manually:

1. OCI Console → **Networking → Virtual Cloud Networks**
2. Click your VCN → **Security Lists** → Default Security List
3. **Add Ingress Rules** — add one rule for each port:

| Source CIDR | Protocol | Port |
|---|---|---|
| 0.0.0.0/0 | TCP | 80 (HTTP) |
| 0.0.0.0/0 | TCP | 443 (HTTPS) |
| 0.0.0.0/0 | TCP | 8000 (FastAPI, optional) |
| 0.0.0.0/0 | TCP | 8501 (Streamlit, optional) |
| 0.0.0.0/0 | TCP | 5000 (MLflow, optional) |

> Ports 8000, 8501, 5000 are already accessible via nginx at `/api/`,
> `/`, and `/mlflow/` on port 80. Direct-port rules are optional.

---

## 3. Run the setup script

One command does everything: installs Docker, clones the repo, trains
the models (~90 minutes on ARM), and starts the full stack.

```bash
ssh ubuntu@YOUR_VM_IP \
  'bash <(curl -fsSL https://raw.githubusercontent.com/dmishra27/scotland-2026-election-forecast/main/deploy/setup_server.sh)'
```

Or copy the script manually and run it:

```bash
scp deploy/setup_server.sh ubuntu@YOUR_VM_IP:~/
ssh ubuntu@YOUR_VM_IP 'bash ~/setup_server.sh'
```

The script will print all working public URLs when it finishes.

---

## 4. Final URLs

| URL | Service |
|---|---|
| `http://VM_IP/` | Streamlit home |
| `http://VM_IP/Voter_Simulator` | Voter simulator |
| `http://VM_IP/Seat_Projections` | Seat projections |
| `http://VM_IP/Model_Performance` | Model performance |
| `http://VM_IP/SHAP_Explainability` | SHAP explainability |
| `http://VM_IP/api/docs` | FastAPI Swagger UI |
| `http://VM_IP/api/health` | API health check |
| `http://VM_IP/mlflow/` | MLflow experiment tracker |

---

## 5. Add a free domain

### Option A — DuckDNS (instant, no account required)

1. Go to **duckdns.org** → sign in with GitHub
2. Add a subdomain, e.g. `scotland-forecast` → point it at `YOUR_VM_IP`
3. You get `scotland-forecast.duckdns.org` for free, forever

### Option B — Cloudflare (professional, free tier)

1. Register a cheap domain (Cloudflare `.com` is ~$10/year)
2. Add an A record pointing to your VM IP
3. Enable the orange cloud (proxied) for free DDoS protection + SSL

---

## 6. Add HTTPS with Let's Encrypt

After your domain is pointing at the VM:

### Step 1 — Stop the nginx container temporarily

```bash
ssh ubuntu@VM_IP
cd ~/scotland-2026-election-forecast
sudo docker compose -f deploy/docker-compose.prod.yml \
  --env-file deploy/.env.prod stop nginx
```

### Step 2 — Run certbot standalone

```bash
sudo certbot certonly --standalone \
  -d scotland-forecast.duckdns.org \
  --agree-tos --non-interactive \
  --email your@email.com
```

### Step 3 — Uncomment the HTTPS server block in nginx.conf

Edit `deploy/nginx.conf` and uncomment the `server { listen 443 ssl; … }`
block. Replace `your-domain.com` with your actual domain.

Also add an HTTP → HTTPS redirect in the port-80 server block:

```nginx
location / {
    return 301 https://$host$request_uri;
}
```

### Step 4 — Restart nginx

```bash
sudo docker compose -f deploy/docker-compose.prod.yml \
  --env-file deploy/.env.prod up -d nginx
```

### Step 5 — Auto-renew certs (add to root crontab)

```bash
sudo crontab -e
# Add:
0 3 * * * /usr/bin/certbot renew --pre-hook \
  "docker compose -f /home/ubuntu/scotland-2026-election-forecast/deploy/docker-compose.prod.yml \
  --env-file /home/ubuntu/scotland-2026-election-forecast/deploy/.env.prod stop nginx" \
  --post-hook \
  "docker compose -f /home/ubuntu/scotland-2026-election-forecast/deploy/docker-compose.prod.yml \
  --env-file /home/ubuntu/scotland-2026-election-forecast/deploy/.env.prod start nginx" \
  --quiet
```

---

## 7. Container status and logs

```bash
# SSH into the server
ssh ubuntu@VM_IP

# Status of all containers
sudo docker compose -f ~/scotland-2026-election-forecast/deploy/docker-compose.prod.yml \
  --env-file ~/scotland-2026-election-forecast/deploy/.env.prod ps

# Tail logs for a specific service
sudo docker compose -f ~/scotland-2026-election-forecast/deploy/docker-compose.prod.yml \
  --env-file ~/scotland-2026-election-forecast/deploy/.env.prod logs -f api

# Quick health checks
curl http://localhost:8000/health
curl http://localhost:8501/_stcore/health
curl http://localhost:5000/health

# Restart everything
sudo docker compose -f ~/scotland-2026-election-forecast/deploy/docker-compose.prod.yml \
  --env-file ~/scotland-2026-election-forecast/deploy/.env.prod restart
```

---

## 8. Continuous deployment (GitHub Actions)

After the initial setup, every push to `main` auto-deploys via
`.github/workflows/deploy.yml`.

Add these **repository secrets** in GitHub → Settings → Secrets:

| Secret | Value |
|---|---|
| `ORACLE_HOST` | Your VM public IP |
| `ORACLE_USER` | `ubuntu` |
| `ORACLE_SSH_KEY` | Contents of your SSH private key (`~/.ssh/id_rsa`) |

The CD pipeline:
1. `git pull origin main`
2. `docker compose up -d --build` (rebuilds only changed layers)
3. Waits 40 s then hits `/health` on all three services

---

## 9. Cost: £0/month forever

| Resource | Free Tier allowance | Used |
|---|---|---|
| Compute | 4 ARM OCPUs + 24 GB | 4 OCPU + 24 GB |
| Block storage | 200 GB | ~20 GB (OS + Docker) |
| Outbound transfer | 10 TB/month | << 1 TB |
| Public IP | 2 free | 1 |

Oracle's Always Free resources never expire and are never charged.
No credit card charges unless you manually upgrade to Pay As You Go.
