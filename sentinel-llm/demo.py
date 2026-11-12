"""
Sentinel-LLM — Interactive demo: classify a log line / email / incident text.
Author: Ashutosh Sahoo, M.Tech AI, NIT Silchar

Usage:
  python demo.py
  python demo.py "Failed password for root from 91.2.3.4 attempt 87 brute force"
  echo "some log line" | python demo.py -

Loads sentinel_lora.pt (the LoRA-finetuned checkpoint). The classifier was
trained on (instruction + input) pairs, exactly like an instruction-tuned
LLM — so the demo performs INSTRUCTION ROUTING: it scores the input under
each task's canonical instruction and reports the best-fitting task frame
with the predicted class + softmax confidence + top-3 alternatives.
(Picking the wrong instruction measurably degrades predictions — the same
prompt-conditioning effect seen in real LLMs.)
"""
import os, sys
import torch
import torch.nn.functional as F
from model import SentinelClassifier, encode_batch
from lora import inject_lora

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TASK_INSTRUCTIONS = {
    "log_triage": "Classify this security log entry. Explain the threat if any.",
    "mitre_mapping": "Map the described adversary behaviour to the correct MITRE ATT&CK technique ID.",
    "phishing_detection": "Determine if this email is phishing or legitimate. Justify briefly.",
    "ioc_extraction": "Extract all Indicators of Compromise (IOCs) from the text and state their types.",
    "cve_explanation": "Explain the CVE in plain language for a junior analyst and give mitigation.",
    "incident_severity": "Assign incident severity (critical/high/medium/low) and justify.",
}

SAMPLES = [
    "web-prod-01 sshd: 142 Failed password attempts for admin from 185.220.101.4 in 5 minutes, authentication failure repeated login attempts brute force",
    "Subject: URGENT Verify your account now! Click http://secure-verify-482.xyz/login to confirm password and credit card immediately urgent prize",
    "fw-edge-01 firewall: PORT SCAN detected from 91.108.5.22 probing ports 22, 80, 443, 3389, 445, 8080 within 10 seconds SYN sweep reconnaissance",
    "Ransomware encrypting production database servers across enterprise, backups deleted, business operations halted, data exfiltration confirmed critical incident outage",
    "app-server-03 nginx: 45.112.9.8 - bob \"GET /index.html HTTP/1.1\" 200 6123 \"-\" \"Mozilla/5.0\"",
    "Threat feed: malicious scanner IP 193.32.162.7 observed in brute force campaign targeting SSH worldwide.",
]

def load_model():
    path = os.path.join(BASE_DIR, "sentinel_lora.pt")
    if not os.path.exists(path):
        raise SystemExit("sentinel_lora.pt not found — run: python data_gen.py && python train.py")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = SentinelClassifier(num_labels=cfg["num_labels"], vocab_size=cfg["vocab_size"],
                               embed_dim=cfg["embed_dim"], hidden_dim=cfg["hidden_dim"])
    inject_lora(model, target_suffixes=("fc1", "fc2"), r=cfg["lora_r"], alpha=cfg["lora_alpha"], dropout=0.0)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["labels"]

def predict_routed(model, labels, text):
    """Score the text under every task instruction; return the best frame."""
    best = None
    for task, instr in TASK_INSTRUCTIONS.items():
        idx, mask = encode_batch([instr + " " + text])
        with torch.no_grad():
            probs = F.softmax(model(idx, mask), dim=-1)[0]
        top = torch.topk(probs, k=min(3, len(labels)))
        cand = {"task": task, "pred": labels[top.indices[0].item()], "conf": top.values[0].item(),
                "alts": [(labels[i.item()], round(v.item(), 4)) for i, v in zip(top.indices, top.values)]}
        if best is None or cand["conf"] > best["conf"]:
            best = cand
    return best

def show(model, labels, text):
    r = predict_routed(model, labels, text)
    print("-" * 72)
    print(f"INPUT           : {text[:160]}")
    print(f"ROUTED TASK     : {r['task']}")
    print(f"PREDICTED CLASS : {r['pred']}")
    print(f"CONFIDENCE      : {r['conf']:.2%}")
    print(f"TOP-3           : {r['alts']}")

def main():
    model, labels = load_model()
    print("=" * 72)
    print("Sentinel-LLM Demo — LoRA-finetuned cybersecurity classifier")
    print(f"Labels ({len(labels)}): {labels}")
    print("=" * 72)
    args = sys.argv[1:]
    if args and args[0] == "-":
        text = sys.stdin.read().strip()
        if text: show(model, labels, text)
    elif args:
        show(model, labels, " ".join(args))
    else:
        print("\nRunning built-in sample inputs (pass your own text as an argument):")
        for s in SAMPLES:
            show(model, labels, s)
        print("-" * 72)
        print("Try: python demo.py \"your log line / email / incident text\"")

if __name__ == "__main__":
    main()
