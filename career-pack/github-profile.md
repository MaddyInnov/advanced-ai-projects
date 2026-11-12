<!-- GitHub profile README draft for github.com/Ashutosh130801
     Usage: create a public repo named "Ashutosh130801" and paste everything below
     the line into its README.md. Results are from each project's results.json,
     measured on that project's built-in evaluation data — see the note under the table. -->

# Ashutosh Sahoo

**AI Engineer · M.Tech, Artificial Intelligence — NIT Silchar (CGPA 8.82)**
LLM fine-tuning · RAG systems · Deep learning research · Privacy-preserving ML

I build AI systems end-to-end and evaluate them honestly: baselines, ablations,
held-out test sets, and limitations written down. Every project below ships with
runnable code and a full thesis-style report (theory, implementation, experiments).

[![LinkedIn](https://img.shields.io/badge/LinkedIn-ashutosh--sahoo--485279222-0A66C2)](https://linkedin.com/in/ashutosh-sahoo-485279222)
[![Email](https://img.shields.io/badge/Email-sahooashutosh239@gmail.com-D14836)](mailto:sahooashutosh239@gmail.com)

![Python](https://img.shields.io/badge/Python-3.12-3776AB) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C) ![NumPy](https://img.shields.io/badge/NumPy-1.26-013243) ![LoRA](https://img.shields.io/badge/LLM%20Fine--Tuning-LoRA%20%2F%20QLoRA-7C3AED) ![RAG](https://img.shields.io/badge/RAG-Hybrid%20Retrieval%20%2B%20Reranking-2B50C8) ![Federated Learning](https://img.shields.io/badge/Federated%20Learning-FedAvg%20%2F%20FedProx-059669) ![Diffusion](https://img.shields.io/badge/Diffusion-DDPM%20%2F%20DDIM-DB2777)

---

## Featured Projects — Advanced AI Portfolio (Nov 2026)

| Project | What it is | Headline result* |
|---|---|---|
| **Sentinel-LLM** | LoRA fine-tuning for cybersecurity threat intelligence — LoRA implemented from scratch in PyTorch, plus a PEFT/QLoRA script for Qwen2.5-0.5B | LoRA matched full fine-tuning at **100% test accuracy** while training **1.86%** of parameters (10,672 of 574,923) |
| **DenseHAR** | Transition-aware **dense** human activity recognition — per-timestep segmentation extending the InceptionKAN encoder | **95.35%** dense accuracy on unseen subjects; **90.63%** in transition windows vs **48.03%** for a fixed-window baseline |
| **AgentForge-RAG** | Agentic RAG with hybrid TF-IDF + BM25 retrieval, reranking, grounded answers and guardrails | **Recall@5 = 100%** on both eval splits; groundedness, abstention and injection-block rates all **100%**; 25/25 tests |
| **DiffusionForge** | Diffusion models from scratch — DDPM + DDIM + classifier-free guidance, conditioned U-Net | DDPM samples at **98.44%** class fidelity; DDIM 50-step at 85.94% fidelity and **6.5× faster** |
| **FedGuard** | Federated learning with differential privacy — FedAvg vs FedProx, non-IID clients, DP-SGD | FedProx recovered non-IID accuracy 87.5% → **88.5%** (centralised 89.0%); DP training held **~84%** at estimated ε ≈ 5–9 |
| **ResolveDesk AI** | A **product**: multi-tenant AI customer-support SaaS — cited KB chat widget, confidence-gated ticketing, ML triage, analytics | Triage classifier **100% intent / 98.3% priority** accuracy; 13/13 tests, 10/10 end-to-end smoke test |

\* *Each result is measured on that project's own built-in evaluation dataset (synthetic /
curated for the purpose, with the protocol documented in its repo and report). They are
pipeline and method results — not claims on public production benchmarks. Where a public
benchmark is claimed, it is named explicitly, as in the research highlight below.*

Each project folder contains the code, its `results.json`, and a designed thesis report
(PDF + Word, 34–45 pages) explaining the theory, every feature and function, the full
experiment protocol, and an interview Q&A learning guide.

## Research Highlight

**InceptionKAN** — *Multi-Scale Temporal Kolmogorov-Arnold Networks for Parameter-Efficient
Human Activity Recognition* (M.Tech thesis work, co-author Ramanujam E; submitted to
*Pervasive and Mobile Computing*).
Multi-scale 1D convolutions (k = 1, 3, 5, 11) + SE attention + an RBF-KAN classifier head,
~238K parameters — **97.25% UCI-HAR · 98.45% PAMAP2 · 95.54% WISDM**.
DenseHAR (above) is the follow-up: dense, transition-aware segmentation with
subject-disjoint (LOSO-style) evaluation.

## Also Built

- **Indecimal-RAG** — agentic RAG assistant over internal policy documents (LangChain, FAISS, tool/function calling; API LLMs benchmarked against a local LLaMA 3B model).
- **RecoverIQ** — debt-collections & recovery intelligence platform (FastAPI, React PWA, Kotlin Android, PostgreSQL; MIS analytics, JWT + TOTP 2FA + RBAC).
- **KriyaDot** — all-in-one AI agent desktop platform · **Office Suite** — 15-app AI office suite · **Sauda** — true-price comparison app (Flutter + FastAPI).

## Background

- 🎓 M.Tech, CSE (Artificial Intelligence) — **NIT Silchar**, CGPA **8.82** · B.Tech CSE, CGPA **8.66**
- 💼 Java Developer Intern — **Synopsys, Pune** · Research Intern — **DRDO**
- 🌍 India-based, immediate joiner; freelance and remote work globally

## Work With Me

- **Freelance / contract:** LLM fine-tuning (LoRA/QLoRA) · RAG chatbots & AI support desks · custom ML/DL development & evaluation
- **Contact:** [LinkedIn](https://linkedin.com/in/ashutosh-sahoo-485279222) · sahooashutosh239@gmail.com
