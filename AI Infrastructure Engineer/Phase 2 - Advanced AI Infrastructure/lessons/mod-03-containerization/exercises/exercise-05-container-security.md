# Exercise 05: Container Security & Supply Chain

**Duration:** 14-16 hours (splits cleanly into three sessions — Part A, Part B, Part C)
**Difficulty:** Intermediate
**Prerequisites:** Docker, Python 3.11+, familiarity with CVEs/CI-CD; Part B additionally needs `cosign`

## Objective

Build a complete container security pipeline in three stages that build directly on each other:

- **Part A** — a `containersec` CLI that scans images for vulnerabilities (Trivy/Grype), generates an SBOM, evaluates a YAML policy, and produces CI-friendly reports.
- **Part B** — extend that pipeline past scanning into supply-chain integrity: sign images with `cosign`, attach SBOM and SLSA provenance attestations, and gate Kubernetes deploys on a Kyverno policy that requires both.
- **Part C** — extend `containersec` again into a full remediation workflow: discover CVEs at scale, triage them, auto-patch what's fixable, document accepted risk, and produce a compliance report.

By the end, one pipeline takes an image from "unscanned" to "scanned, signed, attested, deploy-gated, and continuously remediated."

## Why this matters

Unscanned images are how CVEs like Log4Shell end up in production, and supply-chain attacks (SolarWinds, codecov, npm typosquatting) are now the dominant breach vector for engineering organizations. A mature platform needs all three layers working together: a gate that blocks known-bad images before they ship (Part A), a verifiable chain of custody proving an image came from a trusted build with a traceable SBOM (Part B), and a sustainable process for the CVEs that inevitably show up after deployment — because a naive "fail on CRITICAL" policy blocks all iteration once you account for real-world noise (Part C). Teams that build this once as a reusable CLI plus cluster policy — rather than re-implementing ad hoc `trivy image` calls per repo — get consistent enforcement and a single place to tune policy.

---

## Part A — Vulnerability scanning CLI

### Requirements

1. **Scan images** with Trivy for OS package vulnerabilities, app dependency vulnerabilities, secrets, and misconfigurations.
2. **Generate an SBOM** in CycloneDX and SPDX formats.
3. **Evaluate a security policy** (YAML) — max severity counts, a CVE blocklist, secret tolerance — and return pass/fail.
4. **Produce reports** in JSON (automation), SARIF (GitHub Security tab), and HTML (human review).
5. **Wire it into CI** so a pull request fails when the policy fails, and the SARIF is uploaded for inline annotations.
6. Optionally aggregate a second scanner (Grype) and reconcile/dedupe findings by CVE ID.

### Step-by-step

**1. Scanner wrapper (90 min)** — Install Trivy. Write `src/scanner/trivy.py` wrapping:
```bash
trivy image --format json --quiet myapp:latest
trivy image --format cyclonedx --quiet myapp:latest   # SBOM
trivy image --scanners secret --format json --quiet myapp:latest
```
Parse each `Results[].Vulnerabilities[]` entry into a `Vulnerability` dataclass (`cve_id`, `package_name`, `installed_version`, `fixed_version`, `severity`, `cvss_score`). Roll counts up into a `ScanResult` (critical/high/medium/low totals + SBOM + secrets + misconfigs).

**2. Policy engine (90 min)** — Define `SecurityPolicy` (max_critical, max_high, max_medium, allow_secrets, blocked_cves, required_base_images) loaded from YAML:
```yaml
policies:
  - name: production
    max_critical: 0
    max_high: 0
    blocked_cves: [CVE-2021-44228, CVE-2021-45046]
  - name: development
    max_critical: 5
    max_high: 20
```
`evaluate(scan_result, policy_name)` returns violations (rule, limit, actual) and an overall pass/fail. Use different policy profiles per environment — a zero-tolerance policy is right for prod but blocks all iteration in dev.

**3. Multi-scanner aggregation (60 min, optional)** — Add a Grype wrapper (`grype myapp:latest -o json`) and a `ScannerAggregator` that runs both in parallel (`ThreadPoolExecutor`), dedupes by CVE ID, and reports where the two scanners disagree on severity — a useful signal for triaging false positives.

**4. Reporting (90 min)** — Generate `scan-results.json` (full `ScanResult`), `results.sarif` (SARIF 2.1.0, one `result` per vulnerability with `ruleId` = CVE ID), and `security-report.html` (vulnerability table + severity chart, grouped by package, recommending `max(fixed_version)`).

**5. Fix recommendations (45 min)** — For each vulnerable package, recommend the minimum version that clears every finding for that package. Recommend a safer base image (distroless, `-slim`, newer tag) when the current base accounts for a disproportionate share of findings.

**6. CI integration (60 min)**
```yaml
- uses: aquasecurity/trivy-action@master
  with:
    image-ref: ${{ github.repository }}:${{ github.sha }}
    format: sarif
    output: trivy-results.sarif
- uses: github/codeql-action/upload-sarif@v2
  with: { sarif_file: trivy-results.sarif }
- run: python -m containersec check --image $IMAGE --policy production --fail-on-violation
```

### Validation

- [ ] `containersec scan myapp:latest` returns a `ScanResult` with correct severity counts.
- [ ] `containersec check --policy production` exits non-zero when a CRITICAL CVE is present.
- [ ] SBOM validates against the CycloneDX schema (`cyclonedx-cli validate`).
- [ ] SARIF file uploads cleanly to GitHub's Security tab and shows inline annotations.
- [ ] CI blocks a PR that introduces a blocked CVE, and passes once it's fixed.

### Common pitfalls

- **Scanning the wrong stage** — running Trivy against a multi-stage Dockerfile's builder stage reports vulnerabilities in tools you never ship. Always scan the final built image.
- **Stale vulnerability DB** — Trivy caches its CVE database; `--skip-db-update` in CI without a refresh job silently misses new CVEs. Refresh at least daily.
- **Treating SBOM generation as scanning** — an SBOM lists packages; it does not itself flag vulnerabilities. You still need `grype sbom:sbom.cdx.json` or Trivy against the image.

---

## Part B — SBOM, signing & supply chain

### Requirements

1. **Generate an SBOM** in CycloneDX format using `syft` (Part A's Trivy-based SBOM works too; `syft` is shown here as the industry-standard alternative).
2. **Scan the SBOM** with `grype` for vulnerabilities.
3. **Sign the image** with `cosign` using **keyless** (Sigstore) signatures.
4. **Attach the SBOM as an attestation** to the signed image, and **attach SLSA provenance** documenting how the image was built.
5. **Deploy a Kyverno policy** that requires image signatures from an allowed identity, requires the SBOM attestation, and fails the deploy if the SBOM contains a HIGH/CRITICAL CVE in a runtime path.

### Step-by-step

**1. Install tooling (15 min)**
```bash
brew install syft grype cosign
```

**2. Generate SBOM (15 min)**
```bash
syft ghcr.io/me/iris-api:0.2 -o cyclonedx-json > sbom.cdx.json
syft ghcr.io/me/iris-api:0.2 -o spdx-json     > sbom.spdx.json
```
Use `syft` (or Trivy) against the **runtime** image, not the builder stage — otherwise you report vulns in build tools you don't ship.

**3. Scan SBOM (15 min)**
```bash
grype sbom:sbom.cdx.json --fail-on high
```
This will probably fail at first — that's the lesson. Fix one or two HIGH CVEs by switching base image or pinning a newer dependency version.

**4. Keyless sign (30 min)**
```bash
COSIGN_EXPERIMENTAL=1 cosign sign ghcr.io/me/iris-api:0.2
```
Follow the OIDC flow (browser opens, sign in with GitHub). Verify:
```bash
cosign verify ghcr.io/me/iris-api:0.2 \
  --certificate-identity-regexp '.*me@example.com' \
  --certificate-oidc-issuer https://github.com/login/oauth
```

**5. Attach SBOM attestation (30 min)**
```bash
COSIGN_EXPERIMENTAL=1 cosign attest --predicate sbom.cdx.json --type cyclonedx ghcr.io/me/iris-api:0.2

cosign verify-attestation --type cyclonedx \
  --certificate-identity-regexp '.*me@example.com' \
  --certificate-oidc-issuer https://github.com/login/oauth \
  ghcr.io/me/iris-api:0.2
```

**6. SLSA provenance from CI (45 min)** — GitHub Actions emits SLSA provenance via `slsa-github-generator`:
```yaml
jobs:
  build:
    permissions: { contents: read, packages: write, id-token: write }
    uses: slsa-framework/slsa-github-generator/.github/workflows/generator_container_slsa3.yml@v2.0.0
    with:
      image: ghcr.io/${{ github.repository }}/iris-api
      digest: ${{ needs.docker.outputs.digest }}
      registry-username: ${{ github.actor }}
    secrets: { registry-password: ${{ secrets.GITHUB_TOKEN }} }
```
Then: `cosign verify-attestation --type slsaprovenance ghcr.io/me/iris-api:0.2`.

**7. Kyverno policy in cluster (45 min)** — Install: `helm install kyverno kyverno/kyverno -n kyverno --create-namespace`.
```yaml
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata: { name: verify-signed-and-scanned }
spec:
  validationFailureAction: Enforce
  rules:
    - name: require-signed
      match: { any: [{ resources: { kinds: [Pod] } }] }
      verifyImages:
        - imageReferences: ["ghcr.io/me/iris-api:*"]
          attestors:
            - entries: [{ keyless: { subject: "me@example.com", issuer: "https://github.com/login/oauth" } }]
    - name: require-sbom
      match: { any: [{ resources: { kinds: [Pod] } }] }
      verifyImages:
        - imageReferences: ["ghcr.io/me/iris-api:*"]
          attestations:
            - type: cyclonedx
              attestors:
                - entries: [{ keyless: { subject: "me@example.com", issuer: "https://github.com/login/oauth" } }]
```
Deploy unsigned → Kyverno blocks with a clear error. Deploy signed → passes. Start with `validationFailureAction: Audit` on a new cluster (logs only) and switch to `Enforce` after a quiet week.

### Validation

- [ ] `syft` and `grype` produce clean outputs.
- [ ] `cosign verify` succeeds for your image with your identity.
- [ ] Deploying an unsigned image is blocked by Kyverno with a policy-violation message; a signed image passes.
- [ ] CI produces signed + provenance-attested artifacts on every push.

### Common pitfalls

- **Cosign without OIDC** — keyless mode requires CI to have `id-token: write`; locally it requires a browser.
- **Key vs. keyless signing** — key-based signing requires you to manage and rotate keys; keyless is simpler but depends on Sigstore's Rekor.
- **Audit vs. Enforce** — Audit only logs, Enforce blocks; don't jump straight to Enforce on a cluster you haven't observed yet.

---

## Part C — Vulnerability remediation workflow

This extends `containersec` from Part A with a remediation subsystem: discover CVEs at fleet scale, triage them (real risk vs. noise), patch automatically where possible, document accepted risks, and produce a quarterly compliance report.

### Requirements

1. **Discovery**: nightly scan of all production images (reuse Part A's Trivy wrapper); results written to a structured store (SQLite/PostgreSQL).
2. **Triage rules**: HIGH/CRITICAL with a fix → auto-create PR. HIGH/CRITICAL without a fix → flag for human review. MEDIUM with a public exploit (per CISA KEV) → treat as HIGH. Everything else → log only.
3. **Auto-patching**: dependabot-style PRs bumping base image tags / dep versions, with CI re-scanning to verify the fix landed.
4. **Risk acceptance**: a `.trivyignore`-style file with explicit expiry dates and justifications, validated at PR time.
5. **Quarterly report**: per-team open-vulnerability counts by severity + age + remediation history.

### Step-by-step

**1. Inventory + nightly scan (45 min)**
```python
# scan_all.py — reuses containersec's Trivy wrapper from Part A
import subprocess, json, sqlite3
from datetime import datetime

DB = sqlite3.connect("scans.db")
DB.execute("""
  CREATE TABLE IF NOT EXISTS findings (
    image TEXT, cve TEXT, severity TEXT,
    pkg_name TEXT, pkg_version TEXT, fixed_version TEXT,
    scanned_at TIMESTAMP, in_kev BOOL DEFAULT 0,
    PRIMARY KEY (image, cve, pkg_name)
  )
""")

IMAGES = ["ghcr.io/me/iris-api:latest", "ghcr.io/me/feature-store:latest"]

for image in IMAGES:
    out = subprocess.run(
        ["trivy", "image", "--format", "json", "--quiet", "--severity", "MEDIUM,HIGH,CRITICAL", image],
        capture_output=True, text=True, check=True,
    )
    for r in json.loads(out.stdout).get("Results", []):
        for v in r.get("Vulnerabilities", []):
            DB.execute("INSERT OR REPLACE INTO findings VALUES (?,?,?,?,?,?,?,?)",
                (image, v["VulnerabilityID"], v["Severity"], v["PkgName"],
                 v["InstalledVersion"], v.get("FixedVersion", ""), datetime.utcnow(), 0))
DB.commit()
```
Run as a cron / GitHub Actions schedule.

**2. KEV enrichment (30 min)** — pull CISA's Known Exploited Vulnerabilities catalog daily and flag matching CVEs:
```python
import requests
kev_ids = {v["cveID"] for v in requests.get(
    "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json").json()["vulnerabilities"]}
DB.executemany("UPDATE findings SET in_kev=1 WHERE cve=?", [(c,) for c in kev_ids])
```

**3. Triage queue (45 min)**
```python
def auto_patchable():
    return DB.execute("""SELECT image, cve, pkg_name, pkg_version, fixed_version FROM findings
      WHERE severity IN ('HIGH','CRITICAL') AND fixed_version != ''""").fetchall()

def needs_human():
    return DB.execute("""SELECT image, cve, pkg_name, severity, in_kev FROM findings
      WHERE (severity IN ('HIGH','CRITICAL') AND fixed_version = '')
         OR (severity = 'MEDIUM' AND in_kev = 1)""").fetchall()
```

**4. Auto-PR for patches (45 min)**
```python
import github

def open_patch_pr(image, cve, pkg, fixed_version):
    repo = github.Github(token).get_repo("me/iris-api")
    branch = f"security/auto-{cve}"
    # ... edit requirements.txt to bump pkg to fixed_version ...
    pr = repo.create_pull(
        title=f"security: bump {pkg} to address {cve}",
        body=f"Auto-generated. CVE: {cve}. Severity: HIGH/CRITICAL with available fix.",
        head=branch, base="main",
    )
    pr.add_to_labels("security", "auto-patch")
```
CI runs the same scan on the PR branch; only merges if the CVE is resolved. Coalesce PRs by package to avoid pingponging when successive scanner runs bump the same dep repeatedly.

**5. Risk acceptance file (30 min)**
```
# .trivyignore  — Format: CVE_ID  expires=YYYY-MM-DD  reason="..."
CVE-2023-45673  expires=2026-08-01  reason="Only affects Windows; we ship Linux only"
CVE-2024-12345  expires=2026-06-15  reason="No public exploit; upstream fix in v2.5"
```
Validate at PR time and fail if any entry has expired:
```python
import re
from datetime import date
expired = [m.group(1) for line in open(".trivyignore")
           if (m := re.match(r"(CVE-\S+)\s+expires=(\d{4}-\d{2}-\d{2})", line))
           and date.fromisoformat(m.group(2)) < date.today()]
assert not expired, f"Expired ignores: {expired}"
```

**6. Quarterly report (45 min)**
```python
report = DB.execute("""
  SELECT image, severity, COUNT(*) AS open_count,
         AVG(julianday('now') - julianday(scanned_at)) AS avg_age_days
  FROM findings WHERE cve NOT IN (SELECT cve FROM accepted_ignores)
  GROUP BY image, severity
""").fetchall()
# render as Markdown table + push to a shared doc / Slack
```

### Validation

- [ ] Scan covers all production images and persists to DB.
- [ ] KEV enrichment marks at least one finding.
- [ ] Triage correctly classifies findings into 3 buckets (auto-patch, human, log).
- [ ] At least one auto-patch PR was opened and tested as merge-able.
- [ ] An expired ignore is caught by the validation script.

### Common pitfalls

- **500+ findings, team ignores all of it** — filter to the actionable subset (HIGH/CRITICAL with fix + KEV) before showing engineers. Volume kills attention.
- **Ignoring without expiry** — `.trivyignore` becomes a graveyard of stale accepts. Always require expiry.
- **CI image not the deployment image** — scanning the wrong image gives a false sense of security. Scan exactly what ships.

---

## Deliverables

1. `containersec` CLI with `scan`, `check`, `sbom`, `report`, and `diff` subcommands, plus a `.security-policy.yaml`.
2. SBOM (`sbom.cdx.json`), `security-report.html`, and `results.sarif` for one real image.
3. That image signed in the registry with SBOM + SLSA provenance attestations, and a Kyverno policy in-cluster demonstrated to block unsigned images and allow signed ones.
4. A GitHub Actions workflow covering scan → policy check → sign → attest → SARIF upload on every PR.
5. Nightly scan + findings DB, at least one auto-patch PR, a `.trivyignore` with a justified entry, and a sample quarterly report.
6. `SUPPLY_CHAIN.md` and `REMEDIATION.md` describing your policy, exception process, and how engineers should respond to a security PR.

## Stretch goals

- Add `containersec diff old-scan.json new-scan.json` to report new vs. fixed vulnerabilities between builds; track MTTR per severity over time.
- Generate a Rego/OPA policy from the same `SecurityPolicy` object for Kubernetes admission control, as an alternative to Kyverno.
- Add a Sigstore Rekor lookup script (given an image SHA, retrieve the full signing audit trail) and alert on signatures from identities other than your CI service account.
- Add EPSS scoring so a HIGH with EPSS 0.0001 is correctly deprioritized below a MEDIUM with EPSS 0.4; integrate Dependabot for non-container deps and align triage across both.
- Build a dashboard: time-series of open CVE count by severity, with annotations for major patches.
