# Real-Benchmark Results — All Projects

*Generated 2026-11-13. Every figure below comes from the named project's `results_real.json`, produced in the 2026-11-12/13 real-data programme ordered after the synthetic headlines were challenged. Protocol throughout: baseline config evaluated on the benchmark test set once → improvement candidates screened on validation only (train-carved val or official dev split) → final selected config evaluated on test once. Every attempt, including failures, is logged in each project's `results_real.json`. DenseHAR had already received this treatment (HAPT: 90.38% accuracy / 0.9398 macro F1 vs window baseline 78.91% / 0.7258 — see `densehar/REAL_DATA_RESULTS.md`).*

---

## 1. Sentinel-LLM — MITRE ATT&CK + CVE severity (NVD)

**Datasets.** MITRE ATT&CK official CTI STIX bundle (ATT&CK Terms of Use): 706 active techniques → first-`kill_chain_phase` tactic, 15 tactics; stratified 488 train / 106 val / 112 test. CVE severity from the NVD API 2.0 (public domain): **temporal split** per the research brief (train published 2020–21, val 2023, test 2024), chronological per-class quota sampling, 4,400 CVEs exactly balanced across 4 severity classes; CVE-ID/year tokens stripped from features. The optional phishing task was **skipped** — no verifiable public corpus was obtained and nothing was substituted.

| Task | Synthetic headline | Real baseline (LoRA) | Real FINAL (LoRA, val-selected) | Full FT (same recipe) | TF-IDF+LR reference |
|---|---|---|---|---|---|
| ATT&CK tactic | 100% (191/191, synthetic tasks) | 47.32% acc / 0.334 F1 (r=16, 10,480 params) | **49.11% / 0.364** (r=8, 5,240 params, 0.93%) | 56.25% / 0.422 | 72.32% / 0.651 |
| CVE severity (temporal) | — | 47.20% / 0.466 (r=4) | **48.90% / 0.475** (5,152 params, 0.47%) | 54.40% / 0.536 | 55.90% / 0.547 |

**Improvement log (summary).** Winners: longer training + early stop + label smoothing on both tasks; char n-grams helped CVE only. Drops: IDF pooling, ordinal loss, multi-label head. One invalid run (dropped class weights, trainer bug) flagged in the JSON and superseded. Best CVE val score overall belonged to a component-derive TF-IDF reference (0.634) that was **excluded from test evaluation by the pre-registered LoRA-recipe selection rule** and is documented as the obvious next step.
**What changed in interpretation.** The synthetic 100% does not transfer at all. The project's small from-scratch embedding model is beaten by a plain TF-IDF logistic regression on both real tasks (literature SVM bar for ATT&CK tactics ≈82% / 0.78; CVE text ceiling ≈70% temporal — the LoRA model is below even the TF-IDF reference). What survives honestly: the LoRA-vs-full-FT *gap* story (LoRA within ~6–9 points of full FT at 0.5–1.8% of parameters) and the from-scratch LoRA implementation itself. Phase B gains were small and mostly did not transfer from inner-val to test — recorded as a plateau.
**Runtime:** ~2,436s recorded CPU compute + ~13 min rate-limited data pulls.
**Limitations.** First-phase label for multi-tactic techniques is a documented simplification; MITRE test n=112 and inner-val n=80 are small; CVE "NONE" tier absent from NVD primary metrics.

## 2. AgentForge-RAG — SciFact (BEIR)

**Dataset.** SciFact via the official BEIR distribution (UKP): corpus 5,183 abstracts; test 300 queries / 339 binary qrel rows; train/dev 809 queries used only for tuning/screening. Full corpus, document-level chunks (title+abstract), all test queries.

| Rung | Synthetic/internal headline | Real baseline (test) | Real FINAL (test, single eval) |
|---|---|---|---|
| Naive TF-IDF | — | R@3 0.5727 · R@5 0.6366 · nDCG@10 0.5769 · MRR@10 0.5394 | — |
| Hybrid TF-IDF+BM25 | — | R@3 0.6521 · R@5 0.7043 · nDCG@10 0.6269 · MRR@10 0.5901 | — |
| Hybrid + rerank | R@5 **100%** (28 Qs / 14 self-authored docs) | R@3 0.6711 · R@5 0.7164 · nDCG@10 0.6429 · MRR@10 0.6078 | — |
| Final: BM25(k1=1.0,b=0.5) + TF-IDF title-boost β=0.5, RRF w=0.3, no reranker | — | — | **R@3 0.6578 · R@5 0.7059 · nDCG@10 0.6466 · MRR@10 0.6090** (588 ms/query) |

**Improvement log (summary).** 44 dev-screened attempts. Dev diagnosis: BM25 alone (0.6512) beat the default hybrid (0.6189) — the project's α=0.55 fusion dilutes BM25; project query expansion changed 0/250 dev queries (cyber lexicon, inert on SciFact). Adopted: BM25 k1/b tuning, RRF fusion weights, TF-IDF title-cosine boost. **Negative results confirming the research brief: RM3 pseudo-relevance feedback hurt (0.6085–0.6259) and was not shipped;** BM25F title weighting and Porter stemming (−0.151) also hurt; the project reranker is incompatible with RRF score scales.
**What changed in interpretation.** The final pipeline beats the best baseline rung by only **+0.0037 nDCG@10** and remains **below the published BEIR BM25 bar of 0.665** — reported in the README as a documented plateau just under the bar for this lexical stack. The ablation *ordering* (hybrid > naive) survives; the 100% recall headline does not.
**Limitations.** Groundedness/abstention are **not** benchmarked by SciFact and remain internal-eval-only; dev gains compressed at each stage (slice → full dev → test).

## 3. DiffusionForge — MNIST (16×16)

**Dataset.** MNIST (LeCun, Cortes & Burges) from the TensorFlow mirror, SHA-256 verified; 60,000/10,000. **Downscaled 28×28 → 16×16** so the existing U-Net applies unchanged — stated everywhere; not comparable to 28×28 FID literature and no FID is claimed. Diffusion training subset 20,000; 2,000-image validation carved from train.

**Evaluator gate (research-brief requirement): PASS** — fidelity classifier scores **98.13%** on the held-out real test split, so fidelity numbers are meaningful.

| Metric | Synthetic (shapes) | Real baseline | Real FINAL (val-selected: +12 epochs, no augmentation) |
|---|---|---|---|
| DDPM-300 fidelity (w=1.5) | 98.44% | 98.57% | **100.0%** (70/70) |
| DDIM-50 fidelity (w=1.5) | 85.94% | 77.14% | **84.29%** |
| CFG w=0 / 1.5 / 3.0 (DDIM) | 28.12 / 85.94 / 54.69 | 8.57 / 77.14 / 34.29 | w re-swept on val; 1.5 selected |
| Diversity (DDPM/DDIM vs real 8.977) | 10.76 / 10.30 | 9.333 / 9.453 | 9.364 / 9.525 |
| Test noise-MSE | 0.0666 | 0.0616 | **0.0576** |

**Improvement log (summary).** Screened on a val fidelity proxy only. **Dropped:** EMA 0.999 (val proxy 11% — at ~950 steps the brief's decay lags the live weights), +epochs *with* ±2px roll augmentation (55%), more-data-with-augmentation (58%), linear schedule (76%). **Kept:** +12 epochs clean (val proxy 87%, val MSE 0.05811). The pattern implicates wrap-around roll augmentation (labelled a hypothesis, not an isolated measurement). **Memorisation check for the 100%:** generated samples' nearest-neighbour distances to the training set are at normal inter-image distances (0% below L2 1.0 vs 5.6% for real test images) — novel samples, not copies.
**What changed in interpretation.** The one project where the real DDPM headline matches its synthetic one (98.44% → 100% final) — because the fidelity method, not the data, was the easy part. The DDIM gap (~16 points below DDPM) is real and wider than the brief's estimate; closing it needed test-side configs beyond the agreed budget and was not done.
**Runtime:** ~5,008s main pipeline + ~10 min Phase B/final.
**Limitations.** 70 samples per sampler (documented change from 64); wall-clock sampling times are shared-CPU contention-dependent; width bump not attempted (DDPM saturated; width doesn't target the DDIM gap under val selection).

## 4. FedGuard — MNIST (16×16)

**Dataset.** MNIST, SHA-256 verified, 60,000/10,000; downscaled to **16×16 = 256 inputs** (MLP 256→64→64→10, 21,258 params — adaptation stated everywhere). Stratified 5,000-image val carved from train (55,000 pool); **test = full 10,000 for every row**; 10 clients, Dirichlet α=0.5 non-IID via the project's own partitioner.

| Config | Synthetic | Real baseline (test) | Real FINAL (val-selected, test once) |
|---|---|---|---|
| Centralised | 0.890 | 0.9756 | **0.9756** (baseline retained) |
| FedAvg IID | 0.886 | 0.9620 | **0.9620** |
| FedAvg non-IID | 0.875 | 0.9623 (15r×3E) | **0.9653** (40r×2E) |
| FedProx non-IID | 0.885 (μ=0.1) | 0.9150 (μ=0.1) | **0.9520** (μ=0.01) — still below FedAvg |
| DP-SGD σ=1.2 | 0.841 (ε̂≈5.0) | 0.8373 | **0.9054** @ ε̂ ≈ 5.55 (estimate) |
| DP-SGD σ=0.7 | 0.839 (ε̂≈9.2) | 0.8353 | **0.9083** @ ε̂ ≈ 10.2 (estimate) |

**Privacy–utility frontier** (best val-selected per ε̂ level; ε̂ is the project's simplified RDP **estimate**, never a formal guarantee): ε̂≈3.81 → 0.7898 · ε̂≈5.55 → 0.9054 · ε̂≈10.2 → 0.9083 · ε̂≈15.31 → 0.9218. Net privacy cost at ε̂≈5.55 ≈ 6 points vs the non-private federated best.
**Improvement log (summary).** 19 logged entries. Rounds were the main lever, as the brief predicted (non-IID 15r → 40r). **Negative results: FedAvgM (val 0.9198) and FedAdam (val 0.9498) both lost to plain FedAvg at the screened settings and were never test-evaluated; FedProx is a negative result on this task at every μ (val monotone decreasing in μ).** DP gains came from learning-rate tuning (0.4 → 2.0), not clipping changes (C=0.5 hurt badly, dropped). Some configs dropped for CPU budget with no numbers claimed.
**What changed in interpretation.** Real numbers are *higher* than synthetic almost everywhere — MNIST is simply an easier task than the synthetic tabular problem, which is itself the lesson: the synthetic table's ordering (centralised ≥ IID ≥ non-IID; DP cost ~5 points) survives, its absolute values did not. Protocol deviations are documented in `results_real.json`: DP rows here use the non-IID partition (the synthetic DP rows used IID), DP lot size raised 32 → 256 for measured CPU feasibility with ε̂ recomputed.
**Runtime:** ~6,071s recorded across 19 configurations.
**Limitations.** 16×16 inputs put absolute numbers below 28×28 literature targets; ε̂ is an estimate throughout; per-config runtimes are contention-variable.

## 5. ResolveDesk AI — Bitext customer-support intents

**Dataset.** Bitext customer-support LLM chatbot training dataset, HF id `bitext/Bitext-customer-support-llm-chatbot-training-dataset`, via the HF parquet API. **Licence: CDLA-Sharing-1.0** (quoted from live HF API metadata). 26,872 rows, 27 intents, 11 categories. Bitext's card describes a hybrid corpus produced with its NLP/NLG technology — a public third-party benchmark independent of this project's templates. **Dedup protocol (research-brief trap):** 24,072 unique normalised utterances; 1,318 duplicate groups covering 4,118 rows; 13.79% of naive-split test rows appear verbatim in naive train.

| Result | Synthetic headline | Real, naive split (leaky) | Real, DEDUPED split — headline protocol |
|---|---|---|---|
| Intent — project classifier as-is (baseline) | 100% acc / 1.000 F1 (600 templates) | 94.75% / 0.9473 | **89.9% acc / 0.8608 F1** (95% CI 89.0–90.7) |
| Intent — FINAL (val-selected) | — | — | **97.59% acc / 0.9748 F1** (CI 97.1–98.0) |
| Category (honest 2nd task) | — | — | baseline 86.68% / 0.8238 → final **99.5% / 0.995** |
| Priority | 98.3% acc | **not evaluated — no real labels exist; remains synthetic-only** | — |

**Improvement log (summary).** 15 val-screened attempts (val carved from deduped train; test untouched). Key finding: deduplication itself unbalances the classes (421–997 unique utterances per intent), so inverse class weighting alone lifted val macro-F1 0.848 → 0.9547 — against the brief's expectation of a wash on the (raw-balanced) rows. Then word bigrams → 0.9526, + char 3–5-grams → 0.9734, + 600 epochs → 0.9780, + l2=1e-3 → **0.9784 (selected; last three configs within 0.0004 = plateau, stopped)**. Baseline's total collapse on `cancel_order` (F1 = 0.0) recovered to 0.9586; weakest final intent `change_order` 0.9261. Drops logged: bigrams alone, no-weighting variants, l2=1e-2, label smoothing.
**What changed in interpretation.** ~4.9 points of the naive real number was duplicate leakage; the honest deduped baseline (89.9%) replaced both the synthetic 100% and the leaky 94.75% as the starting point, and the honest final (97.59%) is *close to* but does not claim the synthetic figure. The shipped product model was **not** retrained (Bitext's 27-intent schema ≠ the product's 6 intents); evaluation lives in `eval_bitext.py`, and the product test suite still passes 13/13. Engineering note: the literal dense baseline was replaced by a sparse execution of the identical features/hyper-parameters after an equivalence check reproduced the dense model exactly (prediction agreement 1.0, max weight difference 0.0) — documented in `results_real.json`.
**Runtime:** baseline 63.1s; Phase B screening 750.7s summed; final 288.0s.
**Limitations.** Bitext is vendor-generated (hybrid) data — public and independent, but not organic ticket text; priority prediction has no real-data evaluation anywhere in this project.

---

## Cross-project summary

| Project | Synthetic headline | Real headline (final) | Verdict |
|---|---|---|---|
| Sentinel-LLM | LoRA 100% | ATT&CK 49.11% / CVE 48.90% (LoRA) | Synthetic collapses; TF-IDF ref beats the small model; LoRA-vs-FT gap story survives |
| AgentForge-RAG | Recall@5 100% | SciFact nDCG@10 0.6466, R@5 0.7059 | Ordering survives; below BEIR BM25 bar 0.665 — documented plateau |
| DiffusionForge | DDPM 98.44% / DDIM 85.94% | DDPM **100%** / DDIM 84.29% (MNIST-16) | Real ≈ synthetic; DDIM gap is the honest open problem |
| FedGuard | 0.890 central / 0.839–0.841 DP | 0.9756 central / 0.9054–0.9083 DP | Real *higher* — task difficulty, not method, set the old numbers |
| ResolveDesk AI | intent 100% / priority 98.3% | intent **97.59% / 0.9748** (deduped); priority synthetic-only | Near-synthetic final, but only via a leak-free protocol + Phase B |

**Programme-level lessons.** (1) Synthetic numbers measured the generator, not the method — in both directions (Sentinel collapsed, FedGuard rose). (2) The two leakage traps the brief flagged were real and material: CVE temporal splitting and Bitext duplicates (~5 points). (3) Phase B helped most where the baseline was mis-configured for the data (ResolveDesk class weighting after dedup; FedGuard rounds + DP learning rate) and least where the method itself is the ceiling (AgentForge lexical stack; Sentinel's small embedding model). (4) Every negative result — RM3, FedProx, FedAvgM/FedAdam, EMA-0.999, augmentation — is logged in the per-project `results_real.json` files, not deleted.
