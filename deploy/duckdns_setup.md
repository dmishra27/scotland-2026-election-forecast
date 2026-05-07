# DuckDNS Free Subdomain Setup

DuckDNS provides free dynamic DNS subdomains. Use it to get a stable hostname
(e.g. `scotland.duckdns.org`) pointing to the Oracle Cloud VM at `79.72.78.114`
so that Let's Encrypt can issue a valid TLS certificate.

---

## 1. Create a DuckDNS account and subdomain

1. Go to <https://www.duckdns.org> and sign in with GitHub, Google, or Reddit.
2. In the **Domains** section enter a subdomain name (e.g. `scotland`) and click
   **add domain**.
3. Copy the **token** shown at the top of the page — you will need it below.

---

## 2. Point the subdomain to 79.72.78.114

Either:

- In the DuckDNS dashboard, type `79.72.78.114` into the **current ip** field
  next to your domain and click **update ip**, or
- Use the API URL in step 3 below (it updates and returns `OK`).

---

## 3. Auto-update IP with a cron job on the Oracle VM

Oracle Cloud sometimes reassigns the public IP after a reboot. The cron job
below refreshes DuckDNS every 5 minutes so the record stays current.

```bash
# Replace DOMAIN with your subdomain (without .duckdns.org)
# Replace TOKEN  with the token from your DuckDNS dashboard
DOMAIN=scotland
TOKEN=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

# Add to crontab (run as the VM user, not root)
(crontab -l 2>/dev/null; echo "*/5 * * * * curl -s \"https://www.duckdns.org/update?domains=${DOMAIN}&token=${TOKEN}&ip=\" >> /var/log/duckdns.log 2>&1") | crontab -
```

Verify it is working:

```bash
curl "https://www.duckdns.org/update?domains=scotland&token=TOKEN&ip="
# Expected response: OK
```

---

## 4. Verify DNS propagation

```bash
# Should resolve to 79.72.78.114
nslookup scotland.duckdns.org
dig +short scotland.duckdns.org
```

Once DNS resolves correctly, run the SSL setup script:

```bash
./deploy/enable_ssl.sh
# Enter: scotland.duckdns.org
```

---

## 5. Final URLs after SSL is enabled

| Service           | URL                                      |
|-------------------|------------------------------------------|
| Streamlit         | https://scotland.duckdns.org/            |
| FastAPI (Swagger) | https://scotland.duckdns.org/api/docs    |
| MLflow UI         | https://scotland.duckdns.org/mlflow/     |
| Health check      | https://scotland.duckdns.org/api/health  |
