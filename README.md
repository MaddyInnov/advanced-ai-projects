# Advanced AI Projects
### Ashutosh Sahoo · M.Tech (AI), NIT Silchar · November 2026

Six complete, runnable, independently verified AI/ML projects — five research & development
builds and one product — each with its code, its measured `results.json`, and a designed
thesis-style report (PDF + editable Word + Markdown, 241 pages in total across the six).

**Honesty note that applies to the whole repository:** every headline result below was measured
on that project's own built-in evaluation dataset (synthetic or curated, protocol documented
in its README and report). These are pipeline/method results, not claims on public production
benchmarks — except where a public benchmark is named: **DenseHAR is evaluated on the real
HAPT dataset** (subject-disjoint), and those real-data numbers are its headline.

## At a glance

| # | Project | Headline result | Evaluation scope |
|---|---|---|---|
| 1 | [Sentinel-LLM](sentinel-llm/) | LoRA (r = 16) matched full fine-tuning at **100% test accuracy (191/191)** training **10,672 of 574,923 params (1.86%)** | Built-in 1,200-example instruction set, 27 labels |
| 2 | [DenseHAR](densehar/) | **90.38% per-timestep accuracy / 0.9398 macro F1** vs window baseline 78.91% / 0.7258 | **Real HAPT data**, subject-disjoint, test = 5 unseen subjects |
| 3 | [AgentForge-RAG](agentforge-rag/) | **Recall@3 and Recall@5 = 100%** on main + paraphrase eval splits; guardrails 100% | Built-in curated corpus + 42-question eval |
| 4 | [DiffusionForge](diffusionforge/) | DDPM samples at **98.44% class fidelity**; DDIM-50 at 85.94% fidelity, **6.5× faster** | Synthetic 16×16 shapes, held-out fidelity classifier |
| 5 | [FedGuard](fedguard/) | Centralised **89.0%** · FedAvg IID **88.6%** · non-IID **87.5%** · DP-SGD **83.9–84.1%** at ε ≈ 5–9 | Synthetic client data, 10 clients, Dirichlet α = 0.5 |
| 6 | [ResolveDesk AI](resolvedesk-ai/) ⭐ | Triage **100% intent / 98.3% priority accuracy**; 13/13 tests, 10/10 smoke steps | 120-message held-out test; working SaaS product |

The full master index — per-project detail, run commands, report links — is in
[INDEX.md](INDEX.md).

---

## The Six Projects

### 1. Sentinel-LLM — `sentinel-llm/`
LoRA fine-tuning for cybersecurity threat intelligence: LoRA implemented from scratch in
PyTorch (plus a ready PEFT/QLoRA script for Qwen2.5-0.5B on GPU — documented, not yet run).
**Headline result:** LoRA (r = 16) matched full fine-tuning at **100% test accuracy (191/191)**
while training **10,672 of 574,923 parameters (1.86%)**; per-task MITRE mapping rose
62.5% → 100% over the base model.
**Run it:** `cd sentinel-llm && python train.py` (then `python evaluate.py` / `python demo.py`)
**Report:** [PDF](reports/sentinel-llm-report.pdf) · [Word](reports/sentinel-llm-report.docx) · [Markdown](reports/sentinel-llm-report.md)

### 2. DenseHAR — `densehar/`
Transition-aware **dense** human activity recognition: per-timestep segmentation extending
the InceptionKAN encoder, evaluated on **real data**.
**Headline result (real HAPT data):** on HAPT (real smartphone acc+gyro @ 50 Hz;
subject-disjoint — test = 5 unseen subjects 2, 4, 9, 10, 12), the final system scores
**90.38% per-timestep accuracy / 0.9398 macro F1** vs the window-128 baseline's
**78.91% / 0.7258** under the identical protocol — transition-window accuracy
**0.6093 vs 0.4787**, transition-class timestep accuracy **0.8744**, segmental F1@25
**0.822 vs 0.749**, edit score **84.88 vs 75.52**. Per-class F1: walking 1.000,
upstairs 1.000, downstairs 0.994, laying 0.976, sitting 0.853, standing 0.815.
**Phase 1 → Phase 2:** config A unchanged only *tied* the baseline on real data
(0.7886 vs 0.7891); the win came from validation-screened levers only — physics
augmentation (the biggest), 12-class transition supervision, a 3-seed ensemble and
val-tuned post-processing — while class weights hurt and boundary-weighted CE was a
null result. The earlier synthetic results (95.35%) are retained in `results.json`
as a preliminary pipeline proof only.
**Run it:** `cd densehar && python train.py` (synthetic pipeline; data: `python data_gen.py` first if `data/` is absent; demo: `python demo.py`) · real-data run: `python load_hapt.py && python train_real.py phase1`
**Real-data report:** [REAL_DATA_RESULTS.md](densehar/REAL_DATA_RESULTS.md)
**Report:** [PDF](reports/densehar-report.pdf) · [Word](reports/densehar-report.docx) · [Markdown](reports/densehar-report.md) (thesis report includes a Real-Data Addendum (HAPT) chapter)

### 3. AgentForge-RAG — `agentforge-rag/`
Agentic RAG with hybrid retrieval (TF-IDF + BM25 from scratch, fusion, query expansion),
reranking, grounded cited answers, abstention and prompt-injection/PII guardrails —
zero external dependencies (stdlib + NumPy).
**Headline result:** **Recall@3 and Recall@5 = 100%** on both the main and paraphrase eval
splits (ablation on paraphrases: 85.71% → 92.86% → 100%); groundedness, abstention and
injection-block rates **100%**; 25/25 tests; mean latency 10.56 ms.
**Run it:** `cd agentforge-rag && python3 evaluate.py` (demo: `python3 demo.py` · tests: `python3 tests.py` · API: `python3 api.py 8000`)
**Report:** [PDF](reports/agentforge-rag-report.pdf) · [Word](reports/agentforge-rag-report.docx) · [Markdown](reports/agentforge-rag-report.md)

### 4. DiffusionForge — `diffusionforge/`
Diffusion models from scratch: DDPM (T = 300) and DDIM samplers, a time- and
class-conditioned U-Net (179,281 parameters) and classifier-free guidance.
**Headline result:** DDPM samples at **98.44% class fidelity**; DDIM-50 retained 85.94%
fidelity at **6.5× speed** (13.4 s vs 87.4 s per 64 samples); guidance ablation
w = 0 / 1.5 / 3.0 → 28.12% / 85.94% / 54.69% fidelity.
**Run it:** `cd diffusionforge && python evaluate.py` (full pipeline: `python data_gen.py && python train.py && python sample.py`)
**Report:** [PDF](reports/diffusionforge-report.pdf) · [Word](reports/diffusionforge-report.docx) · [Markdown](reports/diffusionforge-report.md)

### 5. FedGuard — `fedguard/`
Federated learning with differential privacy: FedAvg vs FedProx across 10 clients,
IID and Dirichlet (α = 0.5) non-IID partitions, DP-SGD with simplified RDP ε estimates.
**Headline result:** centralised **89.0%** · FedAvg IID **88.6%** · FedAvg non-IID **87.5%** ·
FedProx non-IID **88.5%** · DP-SGD **83.9–84.1%** at estimated ε ≈ 9.19 (σ = 0.7) and
ε ≈ 5.03 (σ = 1.2) — the privacy–utility trade-off, measured.
**Run it:** `cd fedguard && python experiments.py` (demo: `python demo.py`)
**Report:** [PDF](reports/fedguard-report.pdf) · [Word](reports/fedguard-report.docx) · [Markdown](reports/fedguard-report.md)

### 6. ResolveDesk AI — `resolvedesk-ai/` ⭐ the product
A working multi-tenant **AI customer-support SaaS** for small businesses: knowledge-base
chat widget with cited answers, a confidence gate that creates a ticket instead of guessing,
ML ticket triage, and deflection analytics — packaged as a sellable service
(setup ₹14,999; plans ₹4,999–₹19,999/month; 48-hour onboarding playbook in `BUSINESS.md`).
**Headline result:** triage classifier **100% intent accuracy / 98.3% priority accuracy**
(120-message held-out test); **13/13 tests** and **10/10 end-to-end HTTP smoke steps** pass.
**Run it:** `cd resolvedesk-ai && python3 app.py` (first time: `python3 train_ml.py` once) → http://localhost:8090
**Report:** [PDF](reports/resolvedesk-ai-report.pdf) · [Word](reports/resolvedesk-ai-report.docx) · [Markdown](reports/resolvedesk-ai-report.md)

---

## Thesis Reports — `reports/`

One designed report per project (navy cover, per-project accent colour, headline-metric
panel, "Results at a Glance" chart page, styled tables): **AgentForge-RAG 43 pp ·
FedGuard 45 pp · DenseHAR 45 pp (incl. the Real-Data Addendum (HAPT) chapter) ·
Sentinel-LLM 39 pp · DiffusionForge 35 pp ·
ResolveDesk AI 34 pp — 241 pages total**, each in PDF + DOCX + MD.
Structure per report (see `reports/REPORT_SPEC.md`): theory from first principles,
function-by-function implementation, dataset, experiment protocol, results & analysis,
limitations, production path, and a learning guide with interview Q&A.
Charts used in the reports live in `reports/assets/`.

## Career Pack — `career-pack/`

| File | Contents |
|---|---|
| `resume-bullets.md` | ATS bullets per project with quantified results, a Skills section, two Professional Summaries, and scope footnotes stating exactly what each number was measured on |
| `upwork-pack.md` | Profile titles, overview, 3 service offerings with prices, and proposal templates |
| `github-profile.md` | Ready-to-paste GitHub profile README |
| `linkedin-posts.md` | 4 ready-to-post project announcements |

## Repository layout & data notes

One folder per project (`sentinel-llm/`, `densehar/`, `agentforge-rag/`,
`diffusionforge/`, `fedguard/`, `resolvedesk-ai/`), plus `reports/` and `career-pack/`.

A few large artefacts are **not** in this repository (see `.gitignore`):

- `densehar/data_hapt_raw/` — the raw HAPT dataset (~242 MB, a public UCI dataset; not
  redistributed here). Download it from the UCI Machine Learning Repository and rebuild
  the processed arrays with `densehar/load_hapt.py` (path documented in
  `densehar/REAL_DATA_KAGGLE.md`). The processed HAPT arrays (`densehar/data_hapt.npz`)
  and the synthetic pipeline data (`densehar/data/`, regenerable via
  `densehar/data_gen.py`) *are* included.
- `densehar/data_harth.npz` — processed HARTH arrays (~132 MB), excluded on size;
  regenerable with the DenseHAR data loaders.
- `densehar/.venv-deploy/` — a local Python virtual environment; recreate from the
  project requirements instead.
- `__pycache__/` and `*.pyc` — Python bytecode caches.

## Author

**Ashutosh Sahoo** — M.Tech, Artificial Intelligence, NIT Silchar (CGPA 8.82) · B.Tech CSE (8.66)
GitHub: [github.com/Ashutosh130801](https://github.com/Ashutosh130801) ·
LinkedIn: [linkedin.com/in/ashutosh-sahoo-485279222](https://linkedin.com/in/ashutosh-sahoo-485279222) ·
sahooashutosh239@gmail.com · Immediate joiner, India-based, remote/global freelance.
