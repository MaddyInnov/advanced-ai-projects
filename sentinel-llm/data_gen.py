"""
Sentinel-LLM — Synthetic Cybersecurity Instruction Dataset Generator
Author: Ashutosh Sahoo, M.Tech AI, NIT Silchar

Generates ~1200 instruction examples across 6 tasks:
  - log_triage, mitre_mapping, phishing_detection,
    ioc_extraction, cve_explanation, incident_severity

Each example: {instruction, input, output, task, label}
Seed fixed for reproducibility. Synthetic but realistic templates
with randomised IPs, users, ports, domains, hashes, CVEs.
"""
import json
import random
import hashlib
import os

random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(BASE_DIR, "dataset.jsonl")

USERS = ["admin", "root", "alice", "bob", "charlie", "david", "eve", "john.doe", "svc_backup", "oracle", "postgres", "ubuntu", "ec2-user", "jdoe", "msmith"]
HOSTS = ["web-prod-01", "db-prod-02", "app-server-03", "fw-edge-01", "mail-gw-01", "vpn-gw-02", "k8s-node-07", "dc-01"]
SERVICES = ["sshd", "httpd", "nginx", "postgres", "mysql", "ftp", "rdp", "smb"]

def rand_ip(private=False):
    if private:
        return f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(2,254)}"
    return f"{random.choice([45,91,103,185,193,203,198,77,89])}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(2,254)}"

def rand_domain(malicious=False):
    if malicious:
        words = ["secure-verify", "login-update", "account-confirm", "free-prize", "update-billing", "signin-alert", "verify-now", "bonus-claim"]
        tlds = ["xyz", "top", "click", "buzz", "site", "online"]
        return f"{random.choice(words)}-{random.randint(100,999)}.{random.choice(tlds)}"
    words = ["company", "example", "mybank", "office365", "google", "amazon", "intranet", "corp"]
    return f"{random.choice(words)}.{random.choice(['com','org','net','io'])}"

def rand_hash():
    return hashlib.sha256(str(random.random()).encode()).hexdigest()

def rand_port():
    return random.choice([22, 80, 443, 3389, 21, 25, 53, 445, 8080, 3306, 5432, 135, 139, 993, 8443, 5900, 6379, 27017])

# ---------------------------------------------------------------- log_triage
def gen_log_triage():
    label = random.choice(["benign", "brute_force", "port_scan", "malware_beacon", "sql_injection"])
    src = rand_ip(); dst = rand_ip(private=True); user = random.choice(USERS); host = random.choice(HOSTS)
    hour = f"{random.randint(0,23):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}"
    if label == "benign":
        templates = [
            f"{host} sshd[{random.randint(1000,9999)}]: Accepted password for {user} from {src} port {rand_port()} ssh2 at {hour}",
            f"{host} nginx: {src} - {user} \"GET /index.html HTTP/1.1\" 200 {random.randint(500,9000)} \"-\" \"Mozilla/5.0\"",
            f"{host} systemd: Scheduled backup completed successfully for database prod at {hour}, size {random.randint(100,900)}MB",
            f"{host} kernel: Normal health check passed, CPU {random.randint(10,60)}% MEM {random.randint(20,70)}%",
            f"{host} httpd: {src} \"GET /assets/style.css HTTP/1.1\" 304 0 legitimate user browsing session {user}",
        ]
        inp = random.choice(templates)
        out = "Classification: benign. This is normal legitimate activity with successful authentication or routine system operation. No action required."
    elif label == "brute_force":
        n = random.randint(15, 200)
        templates = [
            f"{host} sshd: {n} Failed password attempts for {user} from {src} in 5 minutes, authentication failure repeated login attempts",
            f"{host} sshd[{random.randint(1000,9999)}]: Failed password for invalid user {user} from {src} port 22 ssh2 - attempt {n} brute force",
            f"{host} ftp: LOGIN FAILURE from {src} user={user} repeated {n} times password guessing attack",
            f"{host} rdp: Multiple failed logon attempts account={user} source={src} count={n} credential stuffing brute force",
        ]
        inp = random.choice(templates)
        out = "Classification: brute_force. Repeated failed authentication attempts from a single source indicate password guessing / credential brute forcing. Block source IP and enforce MFA / account lockout."
    elif label == "port_scan":
        ports = ", ".join(str(p) for p in random.sample([22,80,443,3389,445,8080,3306,21,25,53,135,5900], k=random.randint(5,9)))
        templates = [
            f"{host} firewall: PORT SCAN detected from {src} probing ports {ports} within 10 seconds, SYN sweep reconnaissance",
            f"{host} ids: Nmap scan signature from {src} sequential port probing ports {ports} host discovery scan",
            f"{host} fw-edge: {src} attempted connections to {random.randint(50,500)} distinct ports on {dst} in 60s - port scan reconnaissance",
        ]
        inp = random.choice(templates)
        out = "Classification: port_scan. Sequential / high-volume probing of many ports indicates reconnaissance scanning (e.g. Nmap). Block scanner and review exposed services."
    elif label == "malware_beacon":
        dom = rand_domain(malicious=True); interval = random.choice([30, 60, 120, 300])
        templates = [
            f"{host} dns: Periodic DNS queries to {dom} every {interval}s from {dst}, beaconing pattern C2 communication, DGA domain",
            f"{host} proxy: Regular HTTPS POST to https://{dom}/gate.php every {interval} seconds small payload beacon malware callback",
            f"{host} edr: Process svchost.exe contacting {src}:{random.choice([4444,8080,9001,443])} at fixed {interval}s interval - malware beacon C2 channel jitter low",
        ]
        inp = random.choice(templates)
        out = "Classification: malware_beacon. Fixed-interval outbound communication to a suspicious domain/IP indicates malware command-and-control (C2) beaconing. Isolate host and hunt for persistence."
    else:  # sql_injection
        payloads = [
            "' OR '1'='1' --", "1 UNION SELECT username,password FROM users--",
            "admin'--", "' OR 1=1; DROP TABLE users;--", "1' AND SLEEP(5)--",
            "%27%20OR%20%271%27%3D%271", "id=1 UNION ALL SELECT NULL,version()--",
        ]
        p = random.choice(payloads)
        templates = [
            f"{host} nginx: {src} \"GET /login?id={p} HTTP/1.1\" 500 SQL injection attempt payload detected",
            f"{host} httpd: POST /search q={p} from {src} - SQL syntax error in application log injection attack",
            f"{host} waf: Blocked SQL injection payload \"{p}\" from {src} targeting /product.php UNION SELECT attack",
        ]
        inp = random.choice(templates)
        out = "Classification: sql_injection. Input contains SQL metacharacters / UNION / tautology payload attempting to manipulate backend database query. Block IP, review WAF rules and parameterise queries."
    return {"instruction": "Classify this security log entry. Explain the threat if any.", "input": inp, "output": out, "task": "log_triage", "label": label}

# ---------------------------------------------------------------- mitre_mapping
MITRE = [
    ("T1110", "Brute Force", "Adversary attempts many passwords against SSH admin account credential guessing password spraying"),
    ("T1046", "Network Service Discovery", "Adversary runs Nmap port scan enumerating open services on network hosts reconnaissance discovery"),
    ("T1071", "Application Layer Protocol C2", "Malware uses HTTPS DNS web protocols to communicate with command and control server beacon"),
    ("T1190", "Exploit Public-Facing Application", "Attacker exploits SQL injection vulnerability in public web application to extract database"),
    ("T1566", "Phishing", "Adversary sends spear phishing email with malicious attachment link to steal credentials"),
    ("T1059", "Command and Scripting Interpreter", "Attacker executes PowerShell script shell commands malicious macro scripting interpreter execution"),
]
def gen_mitre():
    code, name, desc = random.choice(MITRE)
    variants = [
        f"Observed behaviour: {desc}. Map to MITRE ATT&CK.",
        f"Incident note: {desc}",
        f"Threat report excerpt: {desc}",
        f"Analyst observation: {desc} — which ATT&CK technique is this?",
    ]
    inp = random.choice(variants)
    out = f"MITRE ATT&CK Technique: {code} - {name}. The described behaviour matches {name}."
    return {"instruction": "Map the described adversary behaviour to the correct MITRE ATT&CK technique ID.", "input": inp, "output": out, "task": "mitre_mapping", "label": code}

# ---------------------------------------------------------------- phishing
def gen_phishing():
    label = random.choice(["phishing", "legitimate"])
    if label == "phishing":
        dom = rand_domain(malicious=True)
        templates = [
            f"Subject: URGENT Verify your account now! Dear customer, your account will be suspended. Click http://{dom}/login to confirm password and credit card immediately. Urgent action required secrecy prize winner.",
            f"Subject: You won a lottery prize! Claim your free prize now click http://{dom} send bank details wire transfer urgent confidential.",
            f"From: support@{dom} Subject: Password expired Reset now at http://{dom}/reset Enter OTP and banking password. Do not share, urgent verify within 24 hours account locked.",
            f"Subject: Invoice overdue payment Attach: invoice.exe Please enable macro and pay via gift card. Urgent CEO request wire transfer confidential phishing credential harvest.",
        ]
        out = "Classification: phishing. Urgency, suspicious domain, credential / payment request and social engineering markers indicate a phishing attempt. Do not click; report and block sender."
    else:
        dom = rand_domain(malicious=False)
        templates = [
            f"Subject: Team meeting notes From: manager@{dom} Hi team, attached are minutes from today's sprint review. Next meeting Thursday 10am in conference room. Regards.",
            f"Subject: Your order has shipped From: orders@{dom} Your order # {random.randint(10000,99999)} is on its way. Track delivery in your account orders page. Thank you for shopping.",
            f"Subject: Monthly newsletter From: news@{dom} This month: product updates, community events and a tutorial on data visualisation. Unsubscribe anytime.",
            f"From: hr@{dom} Subject: Holiday schedule The office will be closed on public holidays listed in the attached HR policy. Payroll runs as usual.",
        ]
        out = "Classification: legitimate. Sender domain, tone and content are consistent with normal business communication with no urgency, credential request or suspicious link."
    inp = random.choice(templates)
    return {"instruction": "Determine if this email is phishing or legitimate. Justify briefly.", "input": inp, "output": out, "task": "phishing_detection", "label": label}

# ---------------------------------------------------------------- ioc_extraction
def gen_ioc():
    ip = rand_ip(); dom = rand_domain(malicious=True); h = rand_hash(); url = f"http://{dom}/payload.exe"
    kind = random.choice(["ip_ioc", "domain_ioc", "hash_ioc", "url_ioc", "multiple_ioc"])
    if kind == "ip_ioc":
        inp = f"Threat feed: malicious scanner IP {ip} observed in brute force campaign targeting SSH worldwide."
        out = f"IOCs extracted: IP address: {ip}. Type: malicious source IP."
    elif kind == "domain_ioc":
        inp = f"Intelligence report: phishing campaign uses domain {dom} to host credential harvesting page."
        out = f"IOCs extracted: Domain: {dom}. Type: phishing domain."
    elif kind == "hash_ioc":
        inp = f"Malware sample SHA256 hash {h} identified as Emotet dropper in sandbox analysis."
        out = f"IOCs extracted: SHA256 file hash: {h}. Type: malware file hash."
    elif kind == "url_ioc":
        inp = f"Blocked URL {url} distributing malware payload via drive-by download."
        out = f"IOCs extracted: URL: {url}. Type: malicious URL."
    else:
        inp = f"Full report: C2 server IP {ip} domain {dom} URL {url} malware hash SHA256 {h} all linked to same campaign."
        out = f"IOCs extracted: IP: {ip}; Domain: {dom}; URL: {url}; SHA256: {h}. Multiple IOC types in single campaign."
    return {"instruction": "Extract all Indicators of Compromise (IOCs) from the text and state their types.", "input": inp, "output": out, "task": "ioc_extraction", "label": kind}

# ---------------------------------------------------------------- cve_explanation
CVES = [
    ("rce", "CVE-2021-44228", "Log4Shell Log4j Java logging library JNDI lookup remote code execution unauthenticated attacker crafted string"),
    ("sqli", "CVE-2023-12345", "SQL injection in web application login form allows attacker to bypass authentication extract database records"),
    ("xss", "CVE-2022-98765", "Cross site scripting XSS stored script injection in comment field executes JavaScript in victim browser session hijack"),
    ("priv_esc", "CVE-2021-3156", "Sudo Baron Samedit heap overflow privilege escalation local user gains root administrator privileges"),
    ("info_disclosure", "CVE-2019-0708", "BlueKeep RDP information disclosure sensitive data exposure? Actually RCE wormable — use info leak: directory traversal exposes confidential files sensitive information disclosure"),
]
# fix info_disclosure description to be clean
CVES[-1] = ("info_disclosure", "CVE-2020-34567", "Directory traversal information disclosure vulnerability exposes sensitive confidential files configuration data to unauthenticated attacker path traversal")
def gen_cve():
    label, cve, desc = random.choice(CVES)
    inp = f"{cve}: {desc}. Explain this vulnerability in plain language and suggest mitigation."
    expl = {
        "rce": "Remote Code Execution: an unauthenticated attacker can run arbitrary code on the server via the vulnerable component. Patch immediately, upgrade Log4j, disable JNDI lookups.",
        "sqli": "SQL Injection: unsanitised input lets an attacker manipulate database queries, bypass login and extract data. Fix with parameterised queries and input validation, patch vendor update.",
        "xss": "Cross-Site Scripting (XSS): injected script runs in victims' browsers, enabling session hijack. Fix by encoding output, CSP headers and patching.",
        "priv_esc": "Privilege Escalation: a low-privileged local user exploits the flaw to gain root/admin. Patch sudo/system component and restrict local access.",
        "info_disclosure": "Information Disclosure / Path Traversal: attacker reads sensitive files/config outside intended directory. Patch, validate paths and restrict file access.",
    }
    out = f"{cve} — {expl[label]}"
    return {"instruction": "Explain the CVE in plain language for a junior analyst and give mitigation.", "input": inp, "output": out, "task": "cve_explanation", "label": label}

# ---------------------------------------------------------------- incident_severity
def gen_severity():
    label = random.choice(["sev_critical", "sev_high", "sev_medium", "sev_low"])
    if label == "sev_critical":
        templates = [
            "Ransomware encrypting production database servers across enterprise, backups deleted, business operations halted, data exfiltration confirmed critical incident outage",
            "Active breach: attacker has domain admin, exfiltrating customer PII database to external server, production compromised critical emergency",
        ]
        out = "Severity: CRITICAL (P1). Active compromise / ransomware with business halt and data exfiltration. Invoke incident response immediately, isolate, executive escalation."
    elif label == "sev_high":
        templates = [
            "Confirmed malware C2 beacon on finance workstation, credential theft suspected, lateral movement detected high severity incident containment needed",
            "Successful brute force login to VPN admin account from foreign IP, unauthorised access confirmed high priority investigation",
        ]
        out = "Severity: HIGH (P2). Confirmed unauthorised access / active malware requiring urgent containment within hours."
    elif label == "sev_medium":
        templates = [
            "Phishing email delivered to 20 employees, 2 clicked link but no credential entry detected, medium severity monitor and reset passwords precaution",
            "Port scan reconnaissance against perimeter firewall from external IP blocked, no exploitation medium severity review logs",
        ]
        out = "Severity: MEDIUM (P3). Potential threat with limited impact, investigation and monitoring required within a day."
    else:
        templates = [
            "Single failed login attempt by employee who forgot password, low severity no threat routine helpdesk ticket",
            "Outdated antivirus signature on one laptop low risk low severity schedule update informational alert no compromise",
        ]
        out = "Severity: LOW (P4). Minimal risk / routine issue, handle in normal operations queue."
    inp = random.choice(templates)
    return {"instruction": "Assign incident severity (critical/high/medium/low) and justify.", "input": inp, "output": out, "task": "incident_severity", "label": label}

GENERATORS = [gen_log_triage, gen_mitre, gen_phishing, gen_ioc, gen_cve, gen_severity]

def main(per_task=200):
    examples = []
    for gen in GENERATORS:
        for _ in range(per_task):
            examples.append(gen())
    random.shuffle(examples)
    with open(OUT_PATH, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    from collections import Counter
    print(f"Wrote {len(examples)} examples to {OUT_PATH}")
    print("Per task:", Counter(e['task'] for e in examples))
    print("Per label:", Counter(e['label'] for e in examples))

if __name__ == "__main__":
    main()
