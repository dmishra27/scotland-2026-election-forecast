#!/usr/bin/env bash
# Run this script on the Oracle Cloud VM AFTER Nginx is up and running.
# It installs certbot, obtains a Let's Encrypt certificate, swaps in
# nginx-ssl.conf, and sets up automatic renewal.
#
# Usage:
#   chmod +x deploy/enable_ssl.sh
#   ./deploy/enable_ssl.sh

set -euo pipefail

# ── 1. Collect domain name ────────────────────────────────────────────────────
read -rp "Enter your domain (e.g. scotland.duckdns.org): " DOMAIN
if [[ -z "$DOMAIN" ]]; then
    echo "Error: domain cannot be empty." >&2
    exit 1
fi
echo "Domain: $DOMAIN"

# ── 2. Install certbot ────────────────────────────────────────────────────────
echo ""
echo "=== Installing certbot ==="
sudo apt-get update -q
sudo apt-get install -y certbot python3-certbot-nginx

# ── 3. Obtain certificate ─────────────────────────────────────────────────────
echo ""
echo "=== Obtaining Let's Encrypt certificate for $DOMAIN ==="
# --nginx plugin edits nginx config automatically; Nginx must be running
sudo certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos \
    --email "admin@$DOMAIN" --redirect

# ── 4. Swap to full SSL config ────────────────────────────────────────────────
APP_DIR="${HOME}/scotland-2026-election-forecast"
echo ""
echo "=== Activating nginx-ssl.conf for $DOMAIN ==="
# Replace the DOMAIN placeholder in the SSL template
sed "s/DOMAIN/$DOMAIN/g" "${APP_DIR}/deploy/nginx-ssl.conf" \
    > /tmp/nginx-ssl-resolved.conf

COMPOSE="docker compose -f ${APP_DIR}/deploy/docker-compose.prod.yml \
         --env-file ${APP_DIR}/deploy/.env.prod"

# Copy resolved config into place
sudo cp /tmp/nginx-ssl-resolved.conf "${APP_DIR}/deploy/nginx.conf"

# Reload nginx inside Docker (avoids a full restart)
${COMPOSE} exec nginx nginx -s reload
echo "Nginx reloaded with SSL config."

# ── 5. Auto-renewal cron job ──────────────────────────────────────────────────
echo ""
echo "=== Setting up auto-renewal cron job ==="
CRON_LINE="0 3 * * * certbot renew --quiet --post-hook 'docker exec \$(docker ps -qf name=nginx) nginx -s reload'"
(crontab -l 2>/dev/null | grep -v "certbot renew"; echo "$CRON_LINE") | crontab -
echo "Cron job added: $CRON_LINE"

# ── 6. Test renewal ───────────────────────────────────────────────────────────
echo ""
echo "=== Testing certificate renewal (dry run) ==="
sudo certbot renew --dry-run
echo "Dry-run renewal succeeded."

# ── 7. Summary ────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " HTTPS is now active for $DOMAIN"
echo "============================================================"
echo " Streamlit dashboard : https://$DOMAIN/"
echo " FastAPI (Swagger)   : https://$DOMAIN/api/docs"
echo " MLflow UI           : https://$DOMAIN/mlflow/"
echo " Health check        : https://$DOMAIN/api/health"
echo "============================================================"
