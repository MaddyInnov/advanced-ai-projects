# Real-Benchmark Improvement Brief — How to Get the Best Honest Results

**Purpose:** the improvement-phase playbook for the five real-benchmark runs (Sentinel-LLM, AgentForge-RAG, DiffusionForge, FedGuard, ResolveDesk). For each project: the published bar, the levers ranked by expected gain × CPU feasibility with concrete implementation notes for our from-scratch codebases, the traps, and a realistic target range (labelled **estimate**).

**Evidence key:** every number below is either (i) quoted from a source with its URL and protocol, or (ii) explicitly labelled **[standard practice]** (textbook/field-standard technique, no specific number claimed) or **[estimate]** (our judgement for planning, not a published result).

**Global rules (apply to all five):**
1. **Evaluation integrity before optimisation.** Fix split hygiene first (dedup, temporal/group splits where the literature demands them, evaluator quality reported). A point gained by leakage is worth less than zero — it will be found.
2. **Select on validation, touch test once per candidate family.** Same rule as the DenseHAR HAPT run.
3. **Cheap levers first.** Most of the available gain in all five projects is in data handling, tuning and post-processing — not in bigger models.
4. **Report the bar's protocol next to our number.** Published numbers are only comparable under matched protocols; where protocols differ, say so in the README (as DenseHAR's report does).

---

## 1. Sentinel-LLM — ATT&CK tactic classification + CVE severity (small embedding model + LoRA vs full FT)

### (a) The published bar
- **ATT&CK tactic tagging (text → tactic):** Crossman et al. (JPMorganChase), *Constructing Multi-label Hierarchical Classification Models for MITRE ATT&CK Text Tagging* — a plain **multiclass SGD-SVM scores 81.95% accuracy / 77.95% F1** on tactic classification of CTI text; their full multi-label hierarchical system reports **~94.55% tactic-level accuracy** (multi-label protocol — *not* the same metric as single-label accuracy; do not compare directly); GPT-4o zero-shot manages only ~59–60%. Sources: https://arxiv.org/abs/2601.14556 and https://arxiv.org/html/2601.14556v1/
- **CVE description → severity tier:** Dixit et al., *Predicting CVSS and CWE from CVE Descriptions with Fine-Tuned Small Language Models* — train 2022–23 CVEs, test on **36,325 CVEs from 2024** (temporal split, deliberately, to stop near-duplicate leakage): a LoRA fine-tuned Phi-3.5-mini (3.8B) reaches **69.9% severity accuracy**; a **TF-IDF logistic-regression baseline lands only ~2.8–4.7 points lower** (≈65–67%); the same LLM zero-shot scores 20.8%. Source: https://arxiv.org/html/2610.24297
- **Reading of the bars:** for tactic classification, a well-built *linear* model is already at ~82% single-label; for CVE severity, ~65–70% is what strong models actually get — severity-from-text is intrinsically noisy (human raters themselves disagree; the task has a hard ceiling well below 90%).

### (b) Ranked levers
1. **Feature upgrade: word n-grams + char n-grams TF-IDF-style features into the embedding/classifier** (if the current pipeline is word-only). **[standard practice]** Security text is full of sub-word signal (CVE IDs patterns excluded — see traps — but `overflow`, `use-after-free`, `cwe-79`-style tokens, version strings). For a from-scratch NumPy/PyTorch model: concatenate hashed char 3–5-gram counts (log1p, L2-normalised) with word features before the classifier head; this is the single biggest cheap win for linear/small models on this kind of text.
2. **Multi-label formulation for ATT&CK.** Techniques genuinely belong to multiple tactics in the STIX data; forcing single-label creates artificial label noise. Predict per-tactic sigmoids, threshold tuned on validation, report both single-label accuracy (argmax, comparable to the 81.95% bar) and multi-label metrics. The 94.55% figure shows how much of the "error" is formulation, not model quality. Implementation: replace softmax head with independent logistic heads — ~10 lines.
3. **Class-imbalance handling for tactics.** ATT&CK tactic distribution is heavily skewed (Discovery/Execution/Defense-Evasion dominate; Resource Development / Recon are rare in technique descriptions). Use per-class weights ∝ 1/sqrt(freq) (gentler than full inverse-frequency — full inverse usually overcorrects **[standard practice]**), tune on validation macro-F1, and always report macro-F1 alongside accuracy (accuracy alone hides rare-tactic collapse).
4. **CVE severity: predict the vector components, derive the tier.** The Stanford study found deriving severity from predicted CVSS components **boosted critical-class recall of TF-IDF LR from 50.7% to 73.3% at no extra cost** (same source as above). If our task is tier-only, adding component prediction as an auxiliary task (or a two-stage predict-then-derive) is the highest-value structural change. If scope must stay tier-only, at minimum use ordinal-aware loss (penalise Low↔Critical confusions more than adjacent-tier ones) **[standard practice]**.
5. **LoRA rank/placement sweep on validation (r ∈ {4, 8, 16}, adapters on all linear layers vs attention-only).** On the synthetic run rank bought confidence, not accuracy; on real noisy labels expect a shallow optimum — pick by validation macro-F1, not by habit. **[standard practice]**
6. **Longer training with early stop on validation + label smoothing 0.05–0.1.** Real security labels are noisy (CVE severity especially); label smoothing is the standard cheap defence against overfitting rater noise. **[standard practice]**

### (c) Traps / negative evidence
- **Random splits on CVE data leak.** Near-duplicate descriptions recur across years/vendors; the Stanford protocol trains on older CVEs and tests on a later year *specifically* "to prevent leakage between near-duplicate descriptions". A random split will inflate severity accuracy by several points and is not comparable to the 69.9% bar. Use a temporal (or at minimum group-by-product/vendor) split.
- **CVE-ID / year tokens as features = retrieval, not classification.** One CVSS study states this explicitly: including the CVE identifier "would compromise the validity of the evaluation". Strip ID/year tokens in preprocessing.
- **Comparing our single-label accuracy to the 94.55% multi-label number** (or vice-versa) — different tasks. Quote the 81.95% SGD-SVM figure for single-label comparisons.
- **Chasing >75% on CVE severity** means overfitting rater idiosyncrasy, not learning security. Treat ~70% as excellence.

### (d) Realistic targets **[estimate]**
- ATT&CK tactic (single-label, held-out techniques): **accuracy 75–85%, macro-F1 0.70–0.80** — at/near the published linear-model bar is a strong result for a small from-scratch model.
- CVE severity (temporal split): **accuracy 60–70%**, with critical-class recall ≥70% via the derive-from-components trick.

---

## 2. AgentForge-RAG — Lexical retrieval on SciFact (BEIR)

### (a) The published bar
- **BM25 on SciFact (BEIR paper, Thakur et al. 2021): nDCG@10 = 0.665**, Recall@100 ≈ 0.908. Confirmed by an independent reproduction study (BM25 66.5, replicated "for most datasets") — https://pure.uva.nl/ws/files/177473152/978-3-031-56066-8_19.pdf — and the BEIR-paper figure is tabulated at https://github.com/ismailperim/briefd/blob/HEAD/eval/beir/README.md
- **Ceiling references:** a *neural* reranker (UPR, T0-3B) on top of BM25 reaches 0.706 (same reproduction paper) — that is **not** reachable by pure lexical means and is not our bar. The best purely-lexical number found: a tuned SQLite-FTS5 BM25 implementation at **0.691** (measured, default settings, no per-dataset tuning claimed) — same briefd README above. Hybrid BM25+dense systems report ~0.69–0.71 (e.g. https://dev.to/neuml/benefits-of-hybrid-search-4fma, BM25 alone 0.663 there) — dense is out of scope for this project.
- **Reading of the bar:** 0.665 is the number to beat; ~0.68–0.70 is the realistic pure-lexical frontier.

### (b) Ranked levers
1. **Get BM25 exactly right first (implementation fidelity).** Most from-scratch BM25s lose points to details, not parameters: correct Robertson IDF (with the +0.5 smoothing and non-negative clamp), document-length normalisation using the true average length, and scoring over the *full* document (title + abstract concatenated). Verify against the bar: if our untuned BM25 is below ~0.64, there is an implementation bug or a tokenisation mismatch — find it before tuning anything. **[standard practice]**
2. **Tokenisation parity with the corpus.** SciFact queries are claims full of biomedical entities; lowercasing + a light stemmer (Porter) and *keeping* hyphenated/compound terms as both split and unsplit tokens typically matters more than k1/b. Index-time: emit both `covid-19` and `covid 19` forms. **[standard practice]**
3. **BM25 parameter sweep on the SciFact *train/dev* queries (never the 300 test queries):** k1 ∈ {0.8, 1.0, 1.2, 1.5}, b ∈ {0.35, 0.5, 0.65, 0.75}. Abstracts are long and length-variance is high, so b below the Lucene default 0.75 usually helps **[standard practice]**; expect this + tokenisation to be worth roughly +0.01–0.03 nDCG@10 over defaults **[estimate]**.
4. **Field weighting (BM25F-style): title repeated/weighted ~2–3× vs abstract.** SciFact titles are dense claim-relevant summaries. Implement as score = BM25(title)·w_t + BM25(abstract)·w_a with (w_t, w_a) tuned on dev. Cheap (two indexes or one index with field tags) and reliably positive on title-bearing corpora **[standard practice]**.
5. **Fuse TF-IDF-cosine and BM25 with reciprocal-rank fusion (RRF, k=60) or CombSUM on normalised scores.** Our hybrid already exists; RRF is the robust default when score scales differ (used throughout the BEIR ecosystem, e.g. https://career.tfks.net/projects/search-lab/notebook). Gains over the better single system are usually small (+0.005–0.02) but consistent **[estimate]**.
6. **Rerank the top-100 with a stronger lexical signal:** e.g. re-score candidates with BM25 on *passage windows* (max-pool over abstract sentences) or with proximity (span) bonuses for query terms occurring close together. One SciFact reproduction found lexical reranking lifted nDCG@10 0.674 → 0.711 in their harness (their BM25 baseline ran hot vs the paper's 0.665, so treat the *delta* ~+0.03 as the signal, not the absolute) — https://github.com/anarvakamdar1/sciverify-scientific-claim-verification-using-rag/blob/HEAD/README.md

### (c) Traps / negative evidence
- **RM3 / pseudo-relevance feedback HURTS on SciFact — twice documented.** The SciVerify reproduction: BM25 nDCG@10 0.674 → **0.522** with KL-divergence PRF, RM3 also below baseline at every cutoff (same README above). An independent 2026 ablation (Appl. Sci. 16:4813) likewise finds "BM25 outperforms BM25 + RM3 across all metrics… RM3 introduces query drift" on claim-style queries — https://www.mdpi.com/2076-3417/16/10/4813 (their absolute numbers run on a different protocol/index — BM25 0.348 — use only the *direction* of the ablation, never the absolutes). SciFact queries are short claims, not keyword queries; expansion terms drift them. **Do not ship RM3 here**; if tried at all, gate it behind a dev-set A/B and expect to reject it.
- **Tuning on the 300 test queries** — with only 300 queries, a k1/b grid fit to test is overfitting with extra steps. Tune on SciFact's dev/train queries, freeze, evaluate test once.
- **Absolute-number comparisons across harnesses are meaningless** (0.348 vs 0.665 for "BM25 on SciFact" in two sources above, due to corpus/protocol differences). Compare deltas within our harness; compare absolutes only to the BEIR paper under the official protocol (full corpus, test queries, title+abstract).

### (d) Realistic target **[estimate]**
**nDCG@10 0.66–0.70** with tuned BM25F + RRF fusion + lexical rerank. Beating 0.665 (the BEIR paper) is publishable-honest; claiming >0.71 lexically should trigger a leakage audit, not a celebration.

---

## 3. DiffusionForge — Conditional DDPM on MNIST @ 16×16, ~180K-param U-Net, classifier-measured fidelity

### (a) The published bar (context, different metric/protocol)
- Published MNIST diffusion quality is usually reported as **FID**, not classifier fidelity: a 4-layer UNet DDPM at 28×28, T=500, 600 epochs reaches **FID 3.67** (conditional variant 2.14) — https://par.nsf.gov/servlets/purl/10550489 ; another benchmark reports DDPM MNIST FID ≈ **11.45** under its protocol — https://arxiv.org/html/2312.08843v1/ . FID at 16×16 with a 180K model and CPU training budgets will be far worse than both; **do not put FID next to these numbers** — our fidelity metric (does a real-data-trained classifier recognise the samples as the requested digit) is the honest yardstick at this scale.
- Directional fidelity reference: a class-conditional DDPM on the harder Fashion-MNIST reached **0.831 conditional classifier accuracy** vs 0.632 for a CVAE in a small course-scale study — https://kyinxu.com/pdfs/image_gen_writeup.pdf (small study, directional only). MNIST is strictly easier, so a well-trained small conditional model should sit well above that.
- EMA evidence: practitioner reference implementation notes sampling *must* use the EMA copy ("the live weights oscillate; the EMA smooths this out", decay 0.9999 typical) — https://www.suchinthaka.io/blog/diffusion-series-3-coding-ddpm ; a score-matching study's ablation table shows FID swinging wildly (spikes >11) without EMA stabilisation — https://sckathach.github.io/assets/dsm_poster.pdf

### (b) Ranked levers
1. **EMA of weights, decay tuned to training length.** At our short CPU runs, 0.9999 is too slow to converge — use **0.999** (or 0.998) so the EMA tracks within ~1–2k steps, and *always sample from EMA weights*. This was already the fix for one of the original synthetic-run bugs (EMA decay mismatch); on real data it remains the single largest fidelity lever per unit effort. **[evidence above + standard practice]**
2. **Training scale: more steps beats more epochs over a subset.** Diffusion fidelity is optimisation-bound at small scale. Prefer the full 60k images for fewer epochs over 20k for many; keep batch 64–128, AdamW lr 2e-4 with cosine decay to ~0, and log fidelity every N steps on a *fixed* noise seed set so the curve (not one lucky sample grid) drives decisions. The reference implementation above reaches legible MNIST digits in ~30 full epochs on GPU; our CPU budget should target the equivalent step count on the downscaled 16×16 task, which is ~3× cheaper per step. **[standard practice]**
3. **Cosine (or at least non-linear) beta schedule + x̂₀ clamping in the sampler.** The project already fixed ε-form drift via posterior-mean sampling with x̂₀ clamping on synthetic data — keep it. A cosine schedule (Nichol & Dhariwal) spends more steps at low noise where digit structure forms; at 16×16 this typically buys visible sharpness for free. **[standard practice]**
4. **Classifier-free guidance: sweep w on validation grids, expect the optimum to move.** On synthetic shapes w=1.5 won and w=3.0 hurt; on MNIST digits, moderate guidance usually helps fidelity at small diversity cost — re-sweep {0, 1, 1.5, 2, 3} *on the fidelity classifier*, and report the diversity metric beside it so the trade-off is explicit, exactly as the original report did. **[standard practice]**
5. **Mild augmentation only: random ±1–2 px translations.** Digits are translation-tolerant; flips/rotations are NOT label-preserving for digits (6↔9) — exclude them. Translation jitter effectively multiplies data and reduces memorisation of the 60k set. **[standard practice]**
6. **DDIM step count: report the fidelity-vs-steps curve (25/50/100).** DDIM-50 was 6.5× faster at −12.5 pts fidelity on synthetic; on the easier MNIST task the gap typically narrows — measure, don't assume. **[standard practice]**

### (c) Traps / negative evidence
- **A weak evaluator inflates fidelity.** If the fidelity classifier is only ~90% accurate on *real* MNIST-16 test data, a "95% fidelity" claim is meaningless. Report the classifier's real-test accuracy in the same table (target: train it properly — it should reach ≥98% at 16×16 **[estimate]** — before trusting it as a judge).
- **FID comparisons across resolutions/feature extractors are invalid** (see (a)). Don't bolt an Inception FID onto 16×16 digits.
- **CFG cherry-picking:** reporting fidelity at the best w without the diversity drop repeats a known inflation pattern; always pair the two.
- **Memorisation masquerading as quality** at small data scales: check nearest-neighbour distances of samples to the training set on a few grids if fidelity looks suspiciously perfect.

### (d) Realistic target **[estimate]**
**Classifier fidelity 90–97%** for DDPM (full-step sampling, EMA weights, moderate CFG), DDIM-50 within ~3–8 points of it, diversity within ~10% of the real-data reference. Below 90% almost always means under-training or an EMA/schedule bug, not a data problem.

---

## 4. FedGuard — FedAvg / FedProx / DP-SGD, MLP on MNIST, 10 clients, Dirichlet α=0.5

### (a) The published bar
- **FedAvg on MNIST is a *solved-scale* problem when rounds are sufficient:** one 2026 benchmark table (Dirichlet α=0.5 partition) reports **FedAvg 98.95% / FedProx 98.40%** final accuracy on MNIST — https://arxiv.org/pdf/2507.23638 (their model/protocol; treat as the "enough-rounds" reference point).
- **Rounds starvation is the dominant failure mode:** a 100-client study capped at **10 rounds** got FedAvg **98.2% IID but only 85.4% at α=0.5**, with FedProx recovering to 88.6% — https://ir.vistas.ac.in/id/eprint/13719/ . Same algorithm, wildly different numbers: rounds and client count explain most cross-paper variance.
- **Server-side adaptive optimisation (FedOpt):** Reddi et al. — FedAdam/FedYogi consistently ≥ FedAvg and much easier to tune; on CIFAR-10, **FedAdam 78.0% vs FedAvg 72.8%** (their benchmark suite) — https://www.alphaxiv.org/abs/2003.00295v5
- **DP-SGD on MNIST (centralised, Abadi et al. 2016):** **97% @ ε=8, 95% @ ε=2, 90% @ ε=0.5** (δ=1e-5), vs 98.3% non-private — PCA+MLP architecture. Summarised at https://www.chriswirz.com/natural-language-processing/deep-learning-with-differential-privacy/index.html and https://www.ccs.neu.edu/home/alina/classes/Fall2024/DP-SGD.pdf . Note: FedGuard's ε is a *simplified RDP estimate* (already labelled as such in the project) — these bars use a proper moments/RDP accountant, so compare accuracy-at-roughly-similar-ε, never ε-for-ε.

### (b) Ranked levers
1. **Rounds first: run to convergence (50–100 rounds at full participation, 10 clients).** With only 10 clients and an MLP on MNIST-16 (256-dim input), CPU cost per round is small; the 85%-vs-99% spread in the literature is mostly this knob. Plot accuracy-vs-round and stop at plateau, not at a round number chosen in advance. **[evidence above]**
2. **Local epochs E: sweep {1, 2, 5} on validation.** More local epochs = fewer rounds needed but more client drift at α=0.5; for an easy task like MNIST, E=2–5 with a slightly lower client LR usually wins on wall-clock. **[standard practice]**
3. **FedAvgM / FedAdam at the server (FedOpt).** Implement server momentum first (5 lines: keep a velocity on the pseudo-gradient Δ = w_old − w_avg, β≈0.9, server lr ~1.0); FedAdam if time permits (~20 lines with m/v state). Evidence (a) shows this is the most reliable post-FedAvg gain and it also *reduces tuning sensitivity* — valuable when every sweep costs CPU hours. **[evidence above]**
4. **FedProx μ: sweep {0.01, 0.1, 1.0} — expect small, sometimes negative, deltas at α=0.5 with enough rounds.** FedProx's clearest wins are under *severe* heterogeneity or rounds starvation (85.4→88.6 in the 10-round study); at convergence on MNIST it can tie or trail FedAvg (98.40 vs 98.95 above). Report it honestly either way — a null FedProx result at α=0.5 with full rounds is consistent with the literature, not a bug. **[evidence above]**
5. **DP-SGD: tune clipping norm before noise.** Per-sample clip C ∈ {0.1, 0.5, 1.0} interacts with everything; with true per-sample clipping (already implemented via torch.func), pick C by validation at fixed σ, use lot size ≥256 (noise averages down), and spend the ε budget on *more steps at moderate σ* rather than few quiet ones. The Abadi bars (97% @ ε≈8) came from exactly this regime (large lots, many steps). **[evidence + standard practice]**
6. **Client fraction = 1.0.** With 10 clients, subsampling only adds variance; full participation is both faster-converging and the honest match to most published MNIST tables. **[standard practice]**

### (c) Traps / negative evidence
- **Comparing across client counts/rounds:** "FedAvg gets 85% on non-IID MNIST" and "FedAvg gets 99%" are both in the literature — the difference is protocol (100 clients × 10 rounds vs 10 clients × converged). Always print rounds/E/client-fraction beside the accuracy.
- **ε estimate vs accountant:** never place FedGuard's simplified-estimate ε on the same axis as Abadi/Opacus ε without the caveat already used in the project docs.
- **Data-order leakage:** reshuffle client partitions with a fixed, documented seed; a lucky α=0.5 draw can swing ±1–2 points. If budget allows, 2 seeds for the headline configs and report the spread **[standard practice]**.
- **DP + FedProx stacking** multiplies hyperparameters; tune DP on the centralised/FedAvg config first, then port.

### (d) Realistic targets **[estimate]**
Centralised MLP ≈ **98–99%**; FedAvg IID ≈ centralised −0–1 pt; **FedAvg α=0.5 ≈ 96–98.5%** at convergence; FedProx within ±0.5 pt of FedAvg; **DP-SGD ≈ 90–95% at ε̂≈5–9** (estimate-labelled ε), i.e. a 3–6 pt privacy cost — consistent with both the original synthetic run's ~4.5-pt cost and the Abadi bars.

---

## 5. ResolveDesk — TF-IDF + softmax LR intent classification on Bitext (27 intents)

### (a) The published bar
- **Bitext is a *templated* dataset, and fine-tuned models saturate it:** a LoRA SFT of an 8B LLM scored **100.0% intent accuracy** on a 270-row stratified held-out sample; zero-shot LLMs score 80–82% intent — https://github.com/hanster-hub/bitext-llm-sft/blob/HEAD/README.md (which also notes *why*: ~2,800-word vocabulary, rigid category↔intent mapping).
- **Classical fine-tuning reference:** Bitext's own GPT-3 fine-tuning study reached **~91–92% intent accuracy with only 100–250 utterances per intent** — https://www.bitext.com/wp-content/uploads/2023/08/FIne-Tuning-GPT-Bitext.pdf
- **Reading of the bar:** a supervised linear model trained on the full ~26.9k rows should be in the high 90s. The scientific risk on Bitext is not accuracy — it's **split leakage** (below).

### (b) Ranked levers
1. **Split hygiene FIRST (this determines whether the number means anything):** Bitext contains many duplicate/near-duplicate utterances. The rigorous protocol used in a 2026 benchmark preprint: *remove duplicate texts, split 80/20 stratified, and drop any test text whose normalised form also occurs in train* — https://www.preprints.org/manuscript/202610.1480 . Implement exactly that (normalise = lowercase, strip punctuation/whitespace) and report **both** the naive random-split number and the deduped number; headline the deduped one.
2. **Word + char n-grams:** word uni+bigrams plus char 3–5-grams (word-boundary aware), sublinear TF (1 + log tf), L2-normalised. Char n-grams are the classic robustness win for user text (typos, inflections) and cost nothing at this corpus size in a hashed/sparse NumPy implementation. **[standard practice — cf. NBSVM-era linear text classification]**
3. **Regularisation sweep:** softmax-LR C (or weight decay) over a log grid {0.1 … 10} on validation; linear text models are usually under-regularised at defaults, and Bitext's small vocabulary makes the sweep cheap. Also try **class-weighted loss** — Bitext intents are near-balanced by construction, so expect weights to be a wash; verify rather than assume. **[standard practice]**
4. **Feature cap sanity:** with ~2,800-word vocabulary, word features alone nearly saturate; if accuracy stalls below ~97% deduped, inspect the confusion pairs (typically `track_order` vs `track_refund`-type neighbours) and add targeted bigrams for the confused pairs rather than growing the model. **[standard practice]**
5. **Train-set size curve:** report accuracy at 25/50/100% of train — Bitext's own study showed saturation by ~250 utterances/intent for a much weaker learner; a flat curve is good evidence the result is data-limited, not model-limited, and pre-empts "just train more" critiques. **[evidence above]**

### (c) Traps / negative evidence
- **Random-split inflation is the #1 trap on this dataset.** With duplicates across the split, a linear model can post ~99–100% that evaporates on deduped data. Any Bitext number without a dedup statement should be assumed inflated — including, potentially, vendor marketing numbers (one vendor claims 99.9% on a "held-out Bitext test set" while estimating 87–93% on real tickets: https://ecomintent.com/ — a useful reminder of the benchmark-to-production gap).
- **Do not import the synthetic priority numbers into the real-data story.** Bitext has no priority labels; the 98.3% priority figure stays labelled synthetic-only (as the report already does).
- **Accuracy is near ceiling — macro-F1 and the confusion matrix are the informative outputs** at this level; a 99.2 vs 99.5 accuracy difference on ~5k test rows is within noise (±~0.3 pt). Report n and, ideally, a Wilson interval. **[standard practice]**

### (d) Realistic target **[estimate]**
**Intent accuracy 97–100% on a deduped stratified split** (expect the naive split to read ~0.5–1.5 pts higher — report both, headline the honest one). Anything below ~95% deduped indicates a pipeline bug (tokenisation/label mapping), not a hard dataset.

---

## Execution order for the improvement phase (all projects)

1. **Integrity pass** — Sentinel: temporal/group CVE split + strip ID tokens; ResolveDesk: dedup protocol; AgentForge: freeze test queries, tune on dev only; DiffusionForge: report evaluator-classifier real-test accuracy; FedGuard: fix/document partition seed.
2. **Free wins** — Sentinel multi-label head + char n-grams; AgentForge tokenisation + BM25F fields; DiffusionForge EMA decay retune + CFG re-sweep; FedGuard rounds-to-plateau + server momentum; ResolveDesk char n-grams + C sweep.
3. **Structural changes only if the bar isn't met** — Sentinel CVSS-component auxiliary task; AgentForge passage-window rerank; FedGuard FedAdam; DiffusionForge cosine schedule.
4. **Explicitly do NOT do** — RM3 on SciFact (documented harm), focal-loss-style reweighting without a validation A/B (documented harm in dense/imbalanced settings in the DenseHAR programme), random-split CVE or Bitext numbers as headlines, FID-at-16×16 comparisons to 28×28 literature.

*Prepared 2026-11-13. All quoted figures carry their source protocol; targets marked [estimate] are planning judgements, not promises — the DenseHAR rule applies: the real number, whatever it is, is the one that gets published.*
