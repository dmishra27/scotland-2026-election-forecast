#!/usr/bin/env bash
# =============================================================================
# Oracle Cloud Free Tier — one-shot server setup
# Tested on: Ubuntu 22.04 ARM64 (VM.Standard.A1.Flex, 4 OCPU, 24 GB RAM)
# Run as the default "ubuntu" user (has passwordless sudo).
# =============================================================================
set -euo pipefail

REPO_URL="https://github.com/dmishra27/scotland-2026-election-forecast"
APP_DIR="${HOME}/scotland-2026-election-forecast"
COMPOSE_FILE="${APP_DIR}/deploy/docker-compose.prod.yml"
ENV_FILE="${APP_DIR}/deploy/.env.prod"

log() { echo -e "\n\033[1;36m>>> $*\033[0m"; }

# ── 1. System packages ────────────────────────────────────────────────────────
log "Installing system packages"
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git curl gnupg ca-certificates lsb-release \
    python3 python3-pip python3-venv \
    nginx certbot python3-certbot-nginx \
    netfilter-persistent iptables-persistent

# ── 2. Docker CE + Compose v2 plugin ─────────────────────────────────────────
log "Installing Docker CE (ARM64)"
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update -y
sudo apt-get install -y \
    docker-ce docker-ce-cli containerd.io \
    docker-buildx-plugin docker-compose-plugin

sudo systemctl enable --now docker
sudo usermod -aG docker "${USER}"
log "Docker $(sudo docker --version)"
log "Compose $(sudo docker compose version)"

# ── 3. Clone repository ───────────────────────────────────────────────────────
log "Cloning repository"
if [ -d "${APP_DIR}/.git" ]; then
    log "Repo already cloned — pulling latest"
    git -C "${APP_DIR}" pull origin main
else
    git clone "${REPO_URL}" "${APP_DIR}"
fi
cd "${APP_DIR}"

# ── 4. Python venv + training dependencies ───────────────────────────────────
log "Creating Python virtual environment"
python3 -m venv .venv
# shellcheck source=/dev/null
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# ── 5. Generate synthetic voter data ─────────────────────────────────────────
log "Generating voter data (12 500 voters)"
python scripts/generate_data.py

# ── 6. Train models (~90 minutes on 4-vCPU ARM) ──────────────────────────────
log "Training stacking ensemble — 20 Optuna trials per base learner"
log "  This takes ~90 minutes. Grab a coffee."
python scripts/train_models.py --n-trials 20
deactivate
log "Training complete. Models in ${APP_DIR}/models/"

# ── 7. Write production .env file ────────────────────────────────────────────
log "Writing deploy/.env.prod"
cat > "${ENV_FILE}" <<EOF
APP_HOME=${APP_DIR}
EOF

# ── 8. Open firewall ports (Oracle requires iptables rules on the instance) ──
log "Configuring iptables"
for PORT in 22 80 443 8000 8501 5000; do
    sudo iptables  -C INPUT -m state --state NEW -p tcp --dport "${PORT}" -j ACCEPT \
        2>/dev/null || \
    sudo iptables  -I INPUT 6 -m state --state NEW -p tcp --dport "${PORT}" -j ACCEPT
done
sudo netfilter-persistent save

# ── 9. Stop the host nginx (Docker nginx will own port 80) ───────────────────
sudo systemctl stop nginx  2>/dev/null || true
sudo systemctl disable nginx 2>/dev/null || true

# ── 10. Build and start Docker stack ─────────────────────────────────────────
log "Building Docker images and starting stack (first build may take 10 min)"
sudo docker compose -f "${COMPOSE_FILE}" --env-file "${ENV_FILE}" up -d --build

# ── 11. Wait for health checks ───────────────────────────────────────────────
log "Waiting for services to become healthy (up to 120 s)"
for i in $(seq 1 24); do
    STATUS=$(sudo docker compose -f "${COMPOSE_FILE}" ps --format json 2>/dev/null \
             | python3 -c "
import sys, json, urllib.request
data = sys.stdin.read().strip()
for line in data.splitlines():
    try:
        s = json.loads(line)
        name = s.get('Name','?')
        state = s.get('Health', s.get('State','?'))
        print(f'  {name}: {state}')
    except Exception:
        pass
" 2>/dev/null || echo "  (waiting...)")
    echo "${STATUS}"
    HEALTHY=$(sudo docker compose -f "${COMPOSE_FILE}" ps 2>/dev/null \
              | grep -c "(healthy)" || true)
    [ "${HEALTHY}" -ge 2 ] && break
    sleep 5
done

# ── 12. Print final URLs ──────────────────────────────────────────────────────
PUBLIC_IP=$(curl -sf --max-time 5 ifconfig.me || curl -sf --max-time 5 icanhazip.com || echo "YOUR_VM_IP")

log "============================================================"
log "  Deployment complete!"
log "============================================================"
echo ""
echo "  Public IP : ${PUBLIC_IP}"
echo ""
echo "  Streamlit home          http://${PUBLIC_IP}/"
echo "  Voter Simulator         http://${PUBLIC_IP}/Voter_Simulator"
echo "  Seat Projections        http://${PUBLIC_IP}/Seat_Projections"
echo "  Model Performance       http://${PUBLIC_IP}/Model_Performance"
echo "  SHAP Explainability     http://${PUBLIC_IP}/SHAP_Explainability"
echo "  FastAPI Swagger         http://${PUBLIC_IP}/api/docs"
echo "  FastAPI health          http://${PUBLIC_IP}/api/health"
echo "  MLflow UI               http://${PUBLIC_IP}/mlflow/"
echo ""
echo "  Check status:  sudo docker compose -f ${COMPOSE_FILE} ps"
echo "  View logs:     sudo docker compose -f ${COMPOSE_FILE} logs -f"
echo ""
log "  NOTE: To use docker without sudo, log out and back in."
log "  NOTE: Open OCI Console → VCN → Security Rules to allow"
log "        ports 80, 443, 8000, 8501, 5000 from 0.0.0.0/0"
