# Resume Bullets — Ashutosh Sahoo
### 6 Advanced AI/ML Projects (November 2026)

**How to use this file:** Paste 2–4 bullets per project under a "Projects" heading. Keep the
scope footnotes in mind — they tell you exactly which claims are safe in an interview and
how to phrase each result if asked "was that on real data?" Every number below is copied
from the project's own results files (`results.json`; DenseHAR from `results_real.json`).

---

## 1. Sentinel-LLM — LoRA Fine-Tuning for Cybersecurity Threat Intelligence

- Implemented LoRA from scratch in PyTorch (low-rank adapters, zero-initialised B matrix, α/r scaling, dropout) and fine-tuned a classifier across 6 cybersecurity tasks — log triage, MITRE ATT&CK mapping, phishing detection, IoC extraction, CVE explanation and incident severity — over a 1,200-example, 27-label instruction dataset.*
- Matched full fine-tuning accuracy (100% on the 191-example held-out test set) while training only 10,672 of 574,923 parameters (1.86%, 52.9× fewer) with the selected rank-16 configuration.*
- Ran a 4-configuration hyperparameter sweep (r = 4 / 8 / 16, two learning rates); all configurations reached 99.44% validation accuracy, with validation loss falling from 0.0475 (r = 4) to 0.0065 (r = 16), and selected the final model on validation loss rather than headline accuracy.*
- Lifted the weakest per-task accuracies from 62.5% → 100% (MITRE mapping), 66.7% → 100% (CVE explanation) and 71.9% → 100% (log triage) over the briefly-trained base model (83.25% test); also authored a PEFT/QLoRA script (4-bit NF4) to fine-tune Qwen2.5-0.5B-Instruct on GPU, documented as ready-to-run.*

## 2. DenseHAR — Transition-Aware Dense Human Activity Recognition

- Built a dense (per-timestep) activity-segmentation network in PyTorch extending the InceptionKAN encoder — multi-scale 1D convolutions (k = 1, 3, 5, 11), SE attention and a dilated temporal stack — at 68,898 parameters, and evaluated it on **real HAPT data** under a strict subject-disjoint protocol (train 21 / validation 4 / test 5 unseen subjects; normalisation statistics from train subjects only; all model selection on validation).*
- Final system — physics-respecting augmentation, 12-class transition supervision, a 3-seed probability ensemble and validation-tuned post-processing — reached **90.38% per-timestep accuracy and 0.9398 macro F1** on the 5 unseen test subjects vs **78.91% / 0.7258** for a window-128 baseline under the identical protocol; transition-window accuracy 0.6093 vs 0.4787 and 87.44% accuracy on the transition-class timesteps themselves.*
- Ran a two-phase, validation-screened improvement programme (22 logged experiments): the phase-1 configuration tied the baseline on real data (78.86% vs 78.91%), after which screening identified physics augmentation as the largest lever (validation F1 0.9189 → 0.9584) — and produced reportable negative results: median-frequency class weights hurt, and boundary-weighted cross-entropy was a null result vs plain cross-entropy.*
- Reported the full MS-TCN-standard segmental suite (boundary F1 0.4087; segmental F1@10/25/50 = 0.836/0.822/0.788; edit score 84.88 vs baseline 75.52) and per-class analysis: dynamic classes solved on unseen subjects (walking and walking-upstairs F1 1.000, downstairs 0.994) with sitting vs standing (0.853 / 0.815) the remaining hard pair. An earlier synthetic-data run (95.35%) is retained only as a preliminary pipeline proof.*

## 3. AgentForge-RAG — Agentic RAG with Hybrid Retrieval, Guardrails & Evaluation

- Built a zero-dependency agentic RAG system in Python/NumPy — Router, Retriever and Verifier agents over a 14-document, 28-chunk policy corpus — with hybrid retrieval (TF-IDF + BM25 implemented from scratch, score fusion, query expansion) and a deterministic reranker.
- Reached Recall@3 and Recall@5 of 100% on both a 28-question eval set and a 14-question paraphrase split; an ablation shows each stage earning its place on paraphrases (naive TF-IDF 85.7% → hybrid 92.9% → hybrid + rerank 100% at Recall@3).*
- Engineered trust guardrails verified end-to-end: 100% groundedness (verifier pass rate, 28/28), 100% correct abstention on unanswerable questions (5/5) and 100% prompt-injection block rate (5/5), with PII redaction on inputs and outputs.
- Tuned chunk size, overlap and the abstention threshold by grid search and shipped an evaluation harness plus 25 passing tests; mean query latency 10.56 ms on the built-in corpus.*

## 4. DiffusionForge — Diffusion Models From Scratch (DDPM + DDIM + Classifier-Free Guidance)

- Implemented the full diffusion pipeline from scratch in PyTorch — linear/cosine noise schedules, closed-form forward noising, a time- and class-conditioned U-Net (179,281 parameters) and classifier-free guidance training (10% label dropout) — with no library diffusion code.
- Trained a DDPM (T = 300, 90 epochs) whose generated samples reached 98.44% class fidelity under an independent checker classifier (99.75% accurate on real data), with sample diversity (mean pairwise L2 10.76) close to real data (11.60).*
- Implemented DDIM sampling and measured the speed/quality trade-off: 50-step DDIM retained 85.94% fidelity at 6.5× the speed of 300-step DDPM (13.4 s vs 87.4 s per 64 samples).*
- Ran a classifier-free-guidance ablation showing guidance is load-bearing at this scale: fidelity 28.12% at w = 0, 85.94% at w = 1.5 and 54.69% at w = 3.0 (over-guidance degrades samples).*

## 5. FedGuard — Federated Learning with Differential Privacy

- Built a federated learning engine in PyTorch implementing FedAvg and FedProx, per-sample-clipped DP-SGD with Gaussian noise, and a simplified RDP-based privacy (ε) estimate, across 10 simulated clients on a 5,000-train / 1,000-test, 10-class dataset.*
- Measured the full comparison grid against a centralised upper bound (89.0%): FedAvg IID 88.6%, FedAvg under Dirichlet (α = 0.5) non-IID label skew 87.5%, and FedProx recovering non-IID accuracy to 88.5%.*
- Quantified the privacy–utility trade-off: DP-SGD training held 83.9–84.1% accuracy at ε ≈ 9.19 (σ = 0.7) and ε ≈ 5.03 (σ = 1.2), δ = 1×10⁻⁵ — roughly 5 accuracy points for a single-digit-ε guarantee, using a compact 6,154-parameter model.*
- Implemented IID and Dirichlet label-skew client partitioning with per-client label histograms, making the non-IID degradation and the FedProx recovery directly observable per round.*

## 6. ResolveDesk AI — AI Customer Support SaaS (Product)

- Designed and shipped a working multi-tenant AI support product (Python stdlib + SQLite + NumPy, zero external APIs): per-business knowledge bases, an embeddable chat widget that answers with citations, and a confidence gate that creates a support ticket instead of guessing when retrieval confidence is low.
- Trained a TF-IDF + softmax-regression triage classifier (NumPy, from scratch) on 600 support messages that classifies every conversation by intent and priority — 100% intent accuracy (macro-F1 1.000) and 98.3% priority accuracy (macro-F1 0.983) on a 120-message held-out test set.*
- Added a hybrid rule + ML escalation layer (explicit urgency signals always override the model), owner dashboard with deflection analytics computed from real database rows, and tenant isolation verified by tests — 13/13 tests and a 10/10 end-to-end HTTP smoke test pass.
- Packaged the product as a sellable service: landing page with pricing (₹4,999–₹19,999/month, ₹14,999 one-time setup), unit-economics model and a 48-hour client-onboarding playbook.

---

## Suggested Updated Skills Section

**Programming:** Python, Java, SQL, JavaScript (basic), Git/GitHub Actions
**LLMs & GenAI:** LoRA/QLoRA fine-tuning (PEFT, from-scratch adapter implementation), instruction tuning, RAG (hybrid TF-IDF + BM25 retrieval, reranking, grounded generation), prompt engineering, tool/function calling, agentic workflows, LLM evaluation & guardrails (groundedness verification, prompt-injection defence, PII redaction)
**Machine Learning / Deep Learning:** PyTorch, NumPy, Pandas, scikit-learn, TensorFlow (working knowledge) — temporal segmentation (MS-TCN-style), diffusion models (DDPM/DDIM, classifier-free guidance), federated learning (FedAvg/FedProx), differential privacy (DP-SGD), hyperparameter tuning, ablation design, subject-disjoint evaluation
**Systems & Product:** FastAPI, REST API design, SQLite/PostgreSQL, WebSockets, multi-tenant SaaS design, ONNX export, Docker (working knowledge), model evaluation harnesses, automated testing

## Professional Summary — (a) AI Engineer Roles

AI Engineer with an M.Tech in Artificial Intelligence from NIT Silchar (CGPA 8.82) and hands-on, end-to-end builds across LLM fine-tuning, RAG systems and production-style ML. Implemented LoRA from scratch and matched full fine-tuning at 1.86% of trainable parameters; built an evaluated agentic RAG system (100% Recall@5 on its eval sets) and shipped a working multi-tenant AI support SaaS. Brings the same discipline to every project: baselines, ablations, held-out evaluation and honestly scoped results. Immediate joiner, open to remote and on-site roles.

## Professional Summary — (b) AI/ML Research-Leaning Roles

M.Tech (AI) researcher from NIT Silchar (CGPA 8.82); co-author of InceptionKAN, a ~238K-parameter multi-scale temporal KAN for human activity recognition (97.25% UCI-HAR, 98.45% PAMAP2, 95.54% WISDM; submitted to *Pervasive and Mobile Computing*). Current research extends that work to transition-aware dense segmentation, evaluated with subject-disjoint splits and segmental metrics, alongside from-scratch implementations of diffusion models (DDPM/DDIM/CFG) and privacy-preserving federated learning (FedAvg/FedProx + DP-SGD). Interested in parameter-efficient, on-device and privacy-aware ML, with reproducible baselines and ablations as standard practice.

---

## \* Scope & Honesty Footnotes — read before using a number in an interview

| Project | Scope of its numbers |
|---|---|
| Sentinel-LLM | Measured on the project's own **synthetic, templated 1,200-example dataset**, which is strongly separable — the 100% proves the LoRA pipeline and the efficiency effect, **not** real-world SOC performance. The local model is a small hashed-embedding classifier proving the pipeline. The QLoRA fine-tune of Qwen2.5-0.5B (`train_lora_peft.py`) is **written and documented but has not been run** — never claim an accuracy for it. |
| DenseHAR | Measured on **real HAPT data** (UCI ML Repository id 341) under a subject-disjoint protocol: train 21 subjects, validation 4, test 5 unseen subjects (2, 4, 9, 10, 12); normalisation from train subjects only; all selection on validation, test evaluated once per final candidate (6 test evaluations total, ledger in `results_real.json`). The earlier synthetic figures (95.35% etc.) are a **pipeline proof only**, superseded as the headline — never present them as benchmark results. InceptionKAN's published numbers (97.25% UCI-HAR etc.) remain a separate windowed result. Be ready to explain the phase-1 tie and the sitting/standing gap; interviewers respect both. |
| AgentForge-RAG | Measured on the project's **built-in corpus and eval sets** (14 documents, 28 + 14 questions). "100%" is scoped to those sets; the abstention threshold margin is narrow (0.2105 vs 0.1919) and is disclosed in the project report. |
| DiffusionForge | Measured on a **synthetic shape-image dataset** generated in code; fidelity is judged by a checker classifier trained on the same distribution. Not an MNIST/CIFAR FID result — no FID is claimed anywhere. |
| FedGuard | Measured on a **synthetic 10-class dataset**; ε values are **simplified RDP estimates**, labelled as estimates in the project — say "estimated ε" if asked. |
| ResolveDesk AI | Triage accuracy is on **synthetic templated messages** (an upper bound); the service model re-tunes the classifier on each client's real tickets. Tests/smoke results (13/13, 10/10) are functional QA on the shipped code, not dataset claims. |

**General rule:** say "on the project's built-in evaluation set" when quoting any figure above.
It costs one second and makes every number defensible.
