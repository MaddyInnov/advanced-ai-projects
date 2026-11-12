"""
Sentinel-LLM — Real benchmark data loaders.
Author: Ashutosh Sahoo (real-data re-run, 2026-11-13)

Loads two REAL public benchmark datasets (no synthetic data):

(a) MITRE ATT&CK CTI STIX bundle (enterprise-attack.json)
    Source : https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json
             (MITRE CTI GitHub repo, https://github.com/mitre/cti)
    Licence: MITRE ATT&CK terms of use — ATT&CK content is freely usable with
             attribution (https://attack.mitre.org/resources/terms-of-use/).
    Task   : technique description -> tactic classification.
             Label = FIRST kill_chain_phase (kill-chain order as listed in the
             STIX object) among phases with kill_chain_name == "mitre-attack".
             145/706 active techniques carry >1 tactic; the full tactic list is
             kept in `all_labels` (used by the Phase-B multi-label evaluation).
             Revoked and x_mitre_deprecated attack-patterns are excluded.
             "(Citation: ...)" parentheticals are stripped from descriptions.

(b) CVE -> CVSS severity, from the NVD API 2.0 (NIST National Vulnerability
    Database; public domain US-government data, https://nvd.nist.gov).
    Task   : English CVE description -> CVSS v3.x base severity tier
             (LOW / MEDIUM / HIGH / CRITICAL; score-0.0 "NONE" records are
             effectively absent from NVD primary v3 metrics — the API's
             cvssV3Severity=NONE filter returns totalResults=0 — so the tier
             set is the four tiers that actually occur).
    Sampling: TEMPORAL protocol (per the brief / Dixit et al. protocol).
             Eras are scanned chronologically through the API; per era we take
             the first records reaching a per-class quota (quota-stop):
               train era 2020-01-01..2022-12-31, quota 700/class
               val   era 2023-01-01..2023-12-31, quota 150/class
               test  era 2024-01-01..2024-12-31, quota 250/class
             If an era is exhausted before a quota is met (LOW is rare), the
             records found are kept and the shortfall is documented in the
             loader summary / results_real.json.
             Only records with a PRIMARY CVSS v3.1 (else v3.0) metric are used;
             the vector components are stored for the Phase-B auxiliary task.
    Leakage hygiene: CVE-ID tokens and standalone year tokens are stripped
             from the description text (they must not be features).

Outputs (project folder):
  dataset_real_mitre.jsonl  {instruction, input, label, all_labels, task, source_id}
  dataset_real_cve.jsonl    {instruction, input, label, task, source_id, year,
                             split_era, cvss_vector, cvss_score}
  real_data/nvd_cache_<era>.jsonl  raw compact NVD records per era (cache)

Usage: python load_real.py [--skip-cve]
"""
import argparse
import json
import os
import re
import time
import urllib.request
import urllib.parse
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "real_data")
MITRE_URL = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
MITRE_PATH = os.path.join(RAW_DIR, "enterprise-attack.json")
NVD_API = "https://services.nvd.nist.gov/rest/json/cves/2.0"

MITRE_INSTRUCTION = "Map the following adversary behaviour to a MITRE ATT&CK tactic."
CVE_INSTRUCTION = "Rate the severity of the following CVE."

_CITATION_RE = re.compile(r"\(Citation:[^)]*\)")
_CVEID_RE = re.compile(r"\bcve[-\s]?\d{4}[-\s]?\d{3,}\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_URL_RE = re.compile(r"https?://\S+")
_WS_RE = re.compile(r"\s+")


def clean_text(t: str) -> str:
    return _WS_RE.sub(" ", t).strip()


# ---------------------------------------------------------------- MITRE
def load_mitre():
    if not os.path.exists(MITRE_PATH):
        os.makedirs(RAW_DIR, exist_ok=True)
        print(f"Downloading MITRE STIX bundle -> {MITRE_PATH}")
        urllib.request.urlretrieve(MITRE_URL, MITRE_PATH)
    bundle = json.load(open(MITRE_PATH))
    rows = []
    n_revoked = n_deprecated = 0
    for o in bundle["objects"]:
        if o.get("type") != "attack-pattern":
            continue
        if o.get("revoked"):
            n_revoked += 1
            continue
        if o.get("x_mitre_deprecated"):
            n_deprecated += 1
            continue
        phases = [p["phase_name"] for p in o.get("kill_chain_phases", [])
                  if p.get("kill_chain_name") == "mitre-attack"]
        desc = o.get("description", "").strip()
        if not phases or not desc:
            continue
        desc = clean_text(_URL_RE.sub(" ", _CITATION_RE.sub(" ", desc)))
        ext = [r for r in o.get("external_references", []) if r.get("source_name") == "mitre-attack"]
        rows.append({
            "instruction": MITRE_INSTRUCTION,
            "input": desc,
            "label": phases[0],                    # first tactic, kill-chain order
            "all_labels": sorted(set(phases)),     # full multi-label ground truth
            "task": "mitre_tactic",
            "source_id": ext[0]["external_id"] if ext else o["id"],
        })
    out = os.path.join(BASE_DIR, "dataset_real_mitre.jsonl")
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary = {
        "source": MITRE_URL,
        "licence": "MITRE ATT&CK Terms of Use (free use with attribution)",
        "n_rows": len(rows),
        "excluded_revoked": n_revoked,
        "excluded_deprecated": n_deprecated,
        "label_rule": "first kill_chain_phase with kill_chain_name='mitre-attack' (kill-chain order)",
        "multi_tactic_rows": sum(1 for r in rows if len(r["all_labels"]) > 1),
        "class_distribution": dict(Counter(r["label"] for r in rows).most_common()),
    }
    print(f"MITRE: {len(rows)} rows -> {out}")
    print(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------- CVE / NVD
ERAS = [
    # (era_name, pubStart, pubEnd, quota_per_class)
    ("train_2020_2022", "2020-01-01T00:00:00.000", "2022-12-31T23:59:59.999", 700),
    ("val_2023",        "2023-01-01T00:00:00.000", "2023-12-31T23:59:59.999", 150),
    ("test_2024",       "2024-01-01T00:00:00.000", "2024-12-31T23:59:59.999", 250),
]
SEVERITIES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def _parse_vector(vs: str):
    parts = {}
    for seg in vs.split("/")[1:]:
        if ":" in seg:
            k, v = seg.split(":", 1)
            parts[k] = v
    return parts


def _extract(cve):
    """Return compact record or None."""
    metrics = cve.get("metrics", {})
    chosen = None
    for key in ("cvssMetricV31", "cvssMetricV30"):
        for m in metrics.get(key, []):
            if m.get("type") == "Primary":
                chosen = m["cvssData"]
                break
        if chosen:
            break
    if not chosen:
        return None
    sev = chosen.get("baseSeverity")
    if sev not in SEVERITIES:
        return None
    desc = ""
    for d in cve.get("descriptions", []):
        if d.get("lang") == "en":
            desc = d["value"]
            break
    if len(desc.split()) < 8:
        return None
    # Leakage hygiene: strip CVE-ID and standalone year tokens.
    text = _YEAR_RE.sub(" ", _CVEID_RE.sub(" ", desc))
    text = clean_text(_URL_RE.sub(" ", text))
    return {
        "instruction": CVE_INSTRUCTION,
        "input": text,
        "label": sev,
        "task": "cve_severity",
        "source_id": cve["id"],
        "year": int(cve["published"][:4]),
        "cvss_score": chosen.get("baseScore"),
        "cvss_vector": _parse_vector(chosen.get("vectorString", "")),
    }


def _nvd_get(params, retries=4):
    url = NVD_API + "?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Sentinel-LLM research (academic)"})
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.load(resp)
        except Exception as e:
            print(f"    NVD request failed ({e}); retry {attempt + 1}/{retries} after backoff", flush=True)
            time.sleep(15 * (attempt + 1))
    raise RuntimeError(f"NVD request failed after {retries} retries: {params}")


def fetch_era(name, start, end, quota):
    """Scan an era chronologically in <=120-day windows (NVD API range limit),
    paginating each window, stopping as soon as every class quota is filled."""
    cache = os.path.join(RAW_DIR, f"nvd_cache_{name}.jsonl")
    if os.path.exists(cache):
        rows = [json.loads(l) for l in open(cache)]
        print(f"NVD era {name}: cache hit, {len(rows)} rows")
        return rows
    from datetime import datetime, timedelta
    fmt = "%Y-%m-%dT%H:%M:%S.%f"
    t0 = datetime.strptime(start, fmt); t1 = datetime.strptime(end, fmt)
    kept, counts = [], Counter()
    pages = 0
    w_start = t0
    while w_start < t1 and not all(counts[s] >= quota for s in SEVERITIES):
        w_end = min(w_start + timedelta(days=119), t1)
        start_index, total = 0, None
        while total is None or start_index < total:
            if all(counts[s] >= quota for s in SEVERITIES):
                break
            params = {"pubStartDate": w_start.strftime(fmt)[:-3], "pubEndDate": w_end.strftime(fmt)[:-3],
                      "resultsPerPage": 2000, "startIndex": start_index}
            payload = _nvd_get(params)
            total = payload["totalResults"]
            pages += 1
            for vuln in payload["vulnerabilities"]:
                rec = _extract(vuln["cve"])
                if rec is None:
                    continue
                if counts[rec["label"]] < quota:
                    rec["split_era"] = name
                    kept.append(rec)
                    counts[rec["label"]] += 1
            print(f"  {name}: window {w_start.date()}..{w_end.date()} page startIndex={start_index} "
                  f"total={total} kept={dict(counts)}", flush=True)
            start_index += 2000
            time.sleep(7)  # NVD unauthenticated rate limit: 5 requests / 30 s
        w_start = w_end + timedelta(milliseconds=1)
    with open(cache, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print(f"NVD era {name}: scanned {pages} pages, kept {len(kept)} -> {cache}")
    return kept


def load_cve():
    all_rows = []
    era_summaries = {}
    for name, start, end, quota in ERAS:
        rows = fetch_era(name, start, end, quota)
        era_summaries[name] = {
            "window": [start[:10], end[:10]], "quota_per_class": quota,
            "n": len(rows), "class_distribution": dict(Counter(r["label"] for r in rows).most_common()),
        }
        all_rows.extend(rows)
    out = os.path.join(BASE_DIR, "dataset_real_cve.jsonl")
    with open(out, "w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    summary = {
        "source": "NVD API 2.0 (https://services.nvd.nist.gov/rest/json/cves/2.0), NIST NVD",
        "licence": "NVD data are public domain (US Government work); CVE descriptions originate from CNAs via the CVE Program",
        "sampling": "temporal eras scanned chronologically, per-class quota-stop; primary CVSS v3.1 else v3.0 metric only",
        "n_rows": len(all_rows),
        "eras": era_summaries,
        "text_hygiene": "CVE-ID tokens and standalone year tokens stripped; URLs stripped",
    }
    print(f"CVE: {len(all_rows)} rows -> {out}")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-cve", action="store_true")
    ap.add_argument("--skip-mitre", action="store_true")
    args = ap.parse_args()
    os.makedirs(RAW_DIR, exist_ok=True)
    summary = {}
    if not args.skip_mitre:
        summary["mitre"] = load_mitre()
    if not args.skip_cve:
        summary["cve"] = load_cve()
    with open(os.path.join(BASE_DIR, "real_data", "provenance.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Provenance -> real_data/provenance.json")
