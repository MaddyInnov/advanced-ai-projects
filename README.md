# Advanced AI Projects
### Ashutosh Sahoo · M.Tech (AI), NIT Silchar · November 2026

Six complete, runnable, independently verified AI/ML projects — five research & development
builds and one product — each with its code, its measured `results.json`, and a designed
thesis-style report (PDF + editable Word + Markdown, 264 pages in total across the six).

**Honesty note that applies to the whole repository:** after the first synthetic headlines were
challenged, **all six projects were re-run on real public benchmark data** — every headline
below is now measured on the named public dataset under the stated protocol (full record in
`REAL_BENCHMARKS.md` and each project's `results_real.json`): DenseHAR on **HAPT**,
Sentinel-LLM on **MITRE ATT&CK + NVD CVE**, AgentForge-RAG on **SciFact (BEIR)**,
DiffusionForge and FedGuard on **MNIST**, ResolveDesk AI on **Bitext**. The earlier
synthetic/internal results are retained in each project as preliminary pipeline proof and
honest contrast — never as headlines — and each report's Real-Data Addendum documents the
baseline, the improvement log (failures included) and the limitations. InceptionKAN's
published benchmark results are listed under Portfolio.

---

## At a glance

| # | Project | Headline result | Evaluation scope |
|---|---|---|---|
| 1 | [Sentinel-LLM](sentinel-llm/) | TF-IDF + linear SVM: **79.46% acc / 0.7788 F1** on ATT&CK tactics (F1 level with the published 0.7795 bar); CVE severity **55.70% / 0.5444** on a temporal split; the from-scratch LoRA model scored 49.11% — the synthetic 100% did not transfer | **Real MITRE ATT&CK + NVD CVE data** |
| 2 | [DenseHAR](densehar/) | **90.38% per-timestep accuracy / 0.9398 macro F1** vs window baseline 78.91% / 0.7258 | **Real HAPT data**, subject-disjoint, test = 5 unseen subjects |
| 3 | [AgentForge-RAG](agentforge-rag/) | SciFact **nDCG@10 0.6466 · Recall@5 0.7059 · MRR@10 0.6090**; ladder 0.5769 → 0.6269 → 0.6429; documented plateau below the published BEIR BM25 bar of 0.665 | **Real SciFact (BEIR) benchmark**, 5,183 abstracts, 300 test queries |
| 4 | [DiffusionForge](diffusionforge/) | DDPM-300 class fidelity **100.0% (70/70)**; DDIM-50 **84.29%**; fidelity judge 98.13% on held-out real test; memorisation check clean | **Real MNIST**, downscaled 28×28 → 16×16, SHA-256 verified |
| 5 | [FedGuard](fedguard/) | Centralised **0.9756** · FedAvg IID **0.9620** · non-IID **0.9653** · DP-SGD **0.9054 @ ε̂ ≈ 5.55 (estimate)**; FedProx a negative result | **Real MNIST** 16×16, full 10,000-image test, 10 clients |
| 6 | [ResolveDesk AI](resolvedesk-ai/) ⭐ | Intent **97.59% acc / 0.9748 F1** · category **99.5% / 0.995** on the deduplicated split (naive 94.75% = leakage contrast only); priority synthetic-only; 13/13 tests | **Real Bitext dataset** (CDLA-Sharing-1.0), deduplicated split |

The full master index — per-project detail, run commands, report links — is in
[INDEX.md](INDEX.md).

---

## The Six Projects

### 1. Sentinel-LLM — `sentinel-llm/`
LoRA fine-tuning for cybersecurity threat intelligence: LoRA implemented from scratch in
PyTorch (plus a ready PEFT/QLoRA script for Qwen2.5-0.5B on GPU — documented, not yet run).
**Headline result (real data — MITRE ATT&CK + NVD CVE):** TF-IDF word 1–2-grams + linear SVM
scores **79.46% accuracy / 0.7788 macro F1** on ATT&CK tactic classification (stratified
split; F1 level with the published SVM bar of 0.7795, accuracy −2.5 pts vs its 81.95% on a
different corpus/protocol) and **55.70% / 0.5444** on CVE severity under a **temporal split**
(train 2020–21, test 2024; literature ceiling ≈70% at 36k-CVE scale). The honest Round 1
story stays on record: the from-scratch LoRA model scored **49.11% / 0.364** (ATT&CK) and
**48.90% / 0.475** (CVE), losing to full fine-tuning within the neural family and to TF-IDF
overall — the synthetic 100% (191/191) did not transfer. Documented next step: the packaged
QLoRA Qwen2.5-0.5B script (not yet run).
**Run it:** `cd sentinel-llm && python train.py` (then `python evaluate.py` / `python demo.py`)
**Report:** [PDF](reports/sentinel-llm-report.pdf) · [Word](reports/sentinel-llm-report.docx) · [Markdown](reports/sentinel-llm-report.md)

### 2. DenseHAR — `densehar/`
Transition-aware **dense** human activity recognition: per-timestep segmentation extending
the InceptionKAN encoder, now evaluated on **real data**.
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
**Headline result (real data — SciFact, BEIR benchmark):** on 5,183 real abstracts with
300 test queries, the final system scores **nDCG@10 0.6466 · Recall@5 0.7059 ·
MRR@10 0.6090**; the ablation ladder survives on real data (TF-IDF 0.5769 → hybrid
0.6269 → +rerank 0.6429). Honest qualifier: after 44 validation-screened attempts the
system plateaus just **below the published BEIR BM25 bar of 0.665** — a documented
plateau, not a win — and RM3 was reproduced as harmful and rejected. Groundedness,
abstention and injection-block rates (100% on the internal self-authored eval; 25/25
tests; 10.56 ms mean latency) are **not** measured by SciFact and stay labelled internal.
**Run it:** `cd agentforge-rag && python3 evaluate.py` (demo: `python3 demo.py` · tests: `python3 tests.py` · API: `python3 api.py 8000`)
**Report:** [PDF](reports/agentforge-rag-report.pdf) · [Word](reports/agentforge-rag-report.docx) · [Markdown](reports/agentforge-rag-report.md)

### 4. DiffusionForge — `diffusionforge/`
Diffusion models from scratch: DDPM (T = 300) and DDIM samplers, a time- and
class-conditioned U-Net (179,281 parameters) and classifier-free guidance.
**Headline result (real data — MNIST, SHA-256 verified, downscaled 28×28 → 16×16):**
final DDPM-300 samples at **100.0% class fidelity (70/70)** and DDIM-50 at **84.29%**,
judged by a fidelity classifier that itself scores **98.13%** on the held-out real test
split; baseline on the same data: DDPM 98.57% / DDIM 77.14%; test noise-MSE **0.0576**;
memorisation check clean (generated samples are novel, not copies). EMA and augmentation
variants are logged as negative results; no FID is claimed and the 16×16 numbers are not
comparable to 28×28 literature. (Synthetic-shapes run: 98.44% DDPM fidelity — preliminary.)
**Run it:** `cd diffusionforge && python evaluate.py` (full pipeline: `python data_gen.py && python train.py && python sample.py`)
**Report:** [PDF](reports/diffusionforge-report.pdf) · [Word](reports/diffusionforge-report.docx) · [Markdown](reports/diffusionforge-report.md)

### 5. FedGuard — `fedguard/`
Federated learning with differential privacy: FedAvg vs FedProx across 10 clients,
IID and Dirichlet (α = 0.5) non-IID partitions, DP-SGD with simplified RDP ε estimates.
**Headline result (real data — MNIST 16×16, full 10,000-image test):** centralised
**0.9756** · FedAvg IID **0.9620** · FedAvg non-IID (Dirichlet α = 0.5, 40 rounds)
**0.9653** · FedProx best **0.9520** — a **negative result**, below FedAvg at every μ ·
DP-SGD **0.9054 at ε̂ ≈ 5.55** (σ = 1.2) and **0.9083 at ε̂ ≈ 10.2** (σ = 0.7), frontier
maximum **0.9218 at ε̂ ≈ 15.31**. Every ε̂ is the project's simplified RDP **estimate**,
never a formal guarantee. (Synthetic run: 89.0% centralised — preliminary.)
**Run it:** `cd fedguard && python experiments.py` (demo: `python demo.py`)
**Report:** [PDF](reports/fedguard-report.pdf) · [Word](reports/fedguard-report.docx) · [Markdown](reports/fedguard-report.md)

### 6. ResolveDesk AI — `resolvedesk-ai/` ⭐ the product
A working multi-tenant **AI customer-support SaaS** for small businesses: knowledge-base
chat widget with cited answers, a confidence gate that creates a ticket instead of guessing,
ML ticket triage, and deflection analytics — packaged as a sellable service
(setup ₹14,999; plans ₹4,999–₹19,999/month; 48-hour onboarding playbook in `BUSINESS.md`).
**Headline result (real data — Bitext customer-support dataset, CDLA-Sharing-1.0):**
on the **deduplicated split**, the triage classifier scores **97.59% intent accuracy /
0.9748 macro F1** and **99.5% / 0.995** on category (deduped baseline: 89.9% / 0.8608).
The naive split's 94.75% is shown only as the leakage contrast — 13.79% of its test rows
appear verbatim in its train set. **Priority prediction has no public labels and remains
synthetic-only.** Product side unchanged: **13/13 tests** and **10/10 end-to-end HTTP
smoke steps** pass; the shipped product model was not retrained (Bitext's 27-intent
schema ≠ the product's 6 intents).
**Run it:** `cd resolvedesk-ai && python3 app.py` (first time: `python3 train_ml.py` once) → http://localhost:8090
**Report:** [PDF](reports/resolvedesk-ai-report.pdf) · [Word](reports/resolvedesk-ai-report.docx) · [Markdown](reports/resolvedesk-ai-report.md)

---

## Thesis Reports — `reports/`

One designed report per project (navy cover, per-project accent colour, headline-metric
panel, "Results at a Glance" chart page, styled tables): **AgentForge-RAG 47 pp ·
FedGuard 50 pp · DenseHAR 45 pp · Sentinel-LLM 44 pp · DiffusionForge 40 pp ·
ResolveDesk AI 38 pp — 264 pages total**, each in PDF + DOCX + MD. **Every report now
carries a Real-Data Addendum chapter and its cover/glance furniture shows the real
benchmark numbers** (DenseHAR: HAPT · Sentinel-LLM App. C: MITRE ATT&CK + NVD CVE ·
AgentForge-RAG App. C: SciFact · DiffusionForge App. C: MNIST · FedGuard App. C: MNIST ·
ResolveDesk AI App. D: Bitext). The combined record is `REAL_BENCHMARKS.md`.
Structure per report (see `reports/REPORT_SPEC.md`): theory from first principles,
function-by-function implementation, dataset, experiment protocol, results & analysis,
limitations, production path, and a learning guide with interview Q&A.
Charts used in the reports live in `reports/assets/`.

## Career Pack — `career-pack/`

| File | Contents |
|---|---|
| `resume-bullets.md` | 3–4 ATS bullets per project with quantified results, an updated Skills section, two 3-line Professional Summaries (AI Engineer / research-leaning), and scope footnotes stating exactly what each number was measured on |
| `upwork-pack.md` | 3 profile title options, a ~250-word profile overview, 3 service offerings with deliverables and starting prices (LLM fine-tuning · ResolveDesk RAG/support setup · custom ML development & evaluation), and 2 proposal templates with [bracketed] personalisation slots |
| `github-profile.md` | Ready-to-paste GitHub profile README: project table with one-line results, InceptionKAN research highlight, skills badges, contact links |
| `linkedin-posts.md` | 5 ready-to-post announcements: the 6-project launch, a Sentinel-LLM deep-dive (why rank bought confidence, not accuracy), a DenseHAR research post (including the over-segmentation counter-result), a DenseHAR real-data follow-up (synthetic headline → phase-1 tie on HAPT → research-driven fixes → verified real numbers), and post (e) — the full six-project real-data story (all projects re-run on public benchmarks, the Sentinel LoRA lesson, the leakage catches, the logged negatives), which supersedes the synthetic headlines in the earlier posts |

## Portfolio

The six projects above sit alongside the existing body of work in the 3D portfolio
(**AI Engineer 3D Portfolio**): **InceptionKAN** (HAR research — 97.25% UCI-HAR /
98.45% PAMAP2 / 95.54% WISDM, ~238K params, submitted to *Pervasive and Mobile Computing*),
**Indecimal-RAG**, **RecoverIQ**, **KriyaDot**, **Office Suite**, **Sauda**, and the
**AI voice-calling platform**.

## Repository layout & data notes

One folder per project (`sentinel-llm/`, `densehar/`, `agentforge-rag/`,
`diffusionforge/`, `fedguard/`, `resolvedesk-ai/`), plus `reports/` and `career-pack/`.

A few large artefacts are **not** in this repository (see `.gitignore` and the loaders):

- `densehar/data_hapt_raw/` — the raw HAPT dataset (a public UCI dataset; not
  redistributed here). Download it from the UCI Machine Learning Repository and rebuild
  the processed arrays with `densehar/load_hapt.py`. The processed HAPT arrays
  (`densehar/data_hapt.npz`) and the synthetic pipeline data (`densehar/data/`,
  regenerable via `densehar/data_gen.py`) *are* included.
- `densehar/data_harth.npz` — processed HARTH arrays (~126 MB), excluded on size;
  regenerable with the DenseHAR data loaders.
- Raw benchmark downloads for the real-data programme — the ATT&CK STIX bundle
  (`sentinel-llm/real_data/`), the SciFact BEIR download (`agentforge-rag/data/`), the
  MNIST archives and derived tensors (`diffusionforge/mnist.npz`,
  `diffusionforge/dataset_real16.pt`, `fedguard/mnist.npz`, `fedguard/mnist16.npz`) and
  the Bitext parquet — are public datasets, fetched by each project's loader/eval
  script (`load_real.py`, `eval_scifact.py`, `eval_bitext.py`, the MNIST loaders),
  which document source, licence and checksums. Small derived samples
  (`sentinel-llm/dataset_real_*.jsonl`) *are* included.
- `densehar/.venv-deploy/` — a local Python virtual environment; recreate from the
  project requirements instead.
- `__pycache__/` and `*.pyc` — Python bytecode caches.

## Author

**Ashutosh Sahoo** — M.Tech, Artificial Intelligence, NIT Silchar (CGPA 8.82) · B.Tech CSE (8.66)
GitHub: [github.com/Ashutosh130801](https://github.com/Ashutosh130801) ·
LinkedIn: [linkedin.com/in/ashutosh-sahoo-485279222](https://linkedin.com/in/ashutosh-sahoo-485279222) ·
sahooashutosh239@gmail.com · Immediate joiner, India-based, remote/global freelance.
