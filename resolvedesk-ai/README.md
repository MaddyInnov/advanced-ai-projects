# ResolveDesk AI — AI Customer Support SaaS (MVP)

A **working product**, not a research demo: a multi-tenant AI customer-support
platform a small business can actually use — and a service its author sells
(setup + monthly retainer; see [BUSINESS.md](BUSINESS.md)).

**Author/owner:** Ashutosh Sahoo · M.Tech (AI), NIT Silchar

## Run it (one command)

```bash
python3 train_ml.py   # once: trains the triage classifier -> results.json + triage_model.npz
python3 app.py        # seeds demo tenants, serves http://localhost:8090
python3 tests.py      # 13 assert-based tests
python3 smoke_test.py # full user journey over real HTTP (10 steps)
```

No external APIs, no API keys, no frameworks: Python stdlib `http.server` +
`sqlite3`, NumPy for ML. Pages: `/` landing · `/dashboard` owner dashboard ·
`/widget-demo` embeddable widget on a fictional store.

**Demo logins (seeded):** `demo@urbankart.in` / `demo1234` (UrbanKart, 10 KB articles) ·
`demo@fitlife.in` / `demo1234` (FitLife Gym, 5 KB articles). Widget keys are listed
in the dashboard and via `GET /api/demo-tenants`.

## Features (every one implemented and tested)
- **Multi-tenancy** — each business registers and gets an isolated KB, tickets,
  conversations and its own embeddable widget key (tests verify tenant isolation).
- **KB management** — owners add/delete help articles via dashboard or API.
- **Cited AI answers** — customer chat retrieves from *that tenant's* KB with
  TF-IDF cosine (stopword removal + light stemming); answers cite the article.
- **No hallucinating** — below a confidence threshold (0.12) the bot does **not**
  guess: it creates a support ticket and tells the customer so.
- **ML triage** — every conversation is classified for **intent** (billing,
  technical, delivery, refund, account, general), **priority** (low/normal/high/
  urgent) and **sentiment** (lexicon). A hybrid rule layer always escalates
  explicit urgency signals ("urgent", "emergency", "ASAP") even when the ML
  head is unsure — standard practice in production triage.
- **Tickets** — auto-created with triage attached, sorted urgent-first, status
  open → in-progress → resolved, owner-scoped updates.
- **Analytics** — total conversations, AI-resolved (deflection) rate, tickets
  created, average confidence, intent breakdown — computed from real DB rows.
- **Owner auth** — salted PBKDF2 password hashing + session tokens.
  ⚠️ *Demo-grade*: no expiry/rate-limiting — see limitations.

## ML triage — REAL benchmark results (`results_real.json`, `eval_bitext.py`)

The classifier (TF-IDF + from-scratch NumPy softmax logistic regression) was
re-evaluated on a **real public benchmark**: the **Bitext customer-support
dataset** on HuggingFace (`bitext/Bitext-customer-support-llm-chatbot-training-dataset`,
**26,872 rows, 27 intents**, licence **CDLA-Sharing-1.0** as stated in the
dataset's HF metadata). Bitext's card describes it as a hybrid corpus generated
with Bitext's NLP/NLG technology — a public third-party benchmark, independent
of this project's own templates.

**Split hygiene first:** Bitext contains many duplicate utterances (24,072
unique normalised texts; 1,318 duplicate groups), so a naive random split leaks
them across train/test — **13.8% of naive test rows appear verbatim in naive
train**. We therefore report both protocols and headline the honest one:
deduplicate on normalised text *before* a stratified 80/20 split (seed 42;
train 19,259 / test 4,813 unique utterances).

### Final real result (headline)

Selected on a validation split carved from the deduped train portion only;
the deduped test split was touched exactly twice (baseline + final).

| Metric (deduped test, n=4,813) | Baseline (project config as-is) | **Final (improved)** |
|---|---|---|
| Intent accuracy | 89.9% (95% CI 89.0–90.7) | **97.59% (95% CI 97.1–98.0)** |
| Intent macro-F1 | 0.861 | **0.975** |
| Category accuracy (2nd task, 11 classes) | 86.7% | **99.5%** |
| Category macro-F1 | 0.824 | **0.995** |

For contrast, the same baseline on the **naive (leaky) split** reads **94.75%**
intent accuracy — ~4.9 points of that is duplicate leakage, which is why the
deduped number is the headline. The baseline's weakest point was dramatic:
`cancel_order` F1 = **0.0** on deduped data (its unique phrasings were absorbed
by `change_order`); the final model scores **0.959** on it, and every intent is
≥ 0.926 F1. Largest remaining confusions: `get_invoice`↔`check_invoice` and
`place_order`→`change_order` (13 test rows each).

**What the improvement phase changed** (all screened on validation only;
full log with keep/drop decisions in `results_real.json`): word uni+bigrams
(20k) + word-boundary char 3–5-grams (30k cap; 19,664 features fitted) TF-IDF,
**inverse class weighting**, l2 = 1e-3, 600 epochs. The surprise lever was
class weighting: raw Bitext rows are balanced, but *deduplication itself*
unbalances the data (unique utterances per intent range 421–997), and fixing
that took validation macro-F1 from 0.848 → 0.955 on its own; char n-grams and
longer training pushed it to the 0.978 plateau (last three configs within
0.0004 — plateau, stopped there). Levers that *hurt* on validation and were
dropped: bigrams or char grams *without* class weighting, l2 = 1e-2, label
smoothing. Train-size curve (validation): 25% of train → 96.8% accuracy,
50% → 97.5%, so the result is close to data-saturated.

Classifier config note: baseline training used `ml.py`'s exact features and
full-batch GD hyper-parameters (max_vocab 3000, epochs 300, lr 0.5, l2 1e-4);
execution ran on the sparse TF-IDF matrix after an equivalence check on the
project's synthetic split reproduced the dense `ml.SoftmaxLR` predictions
100% with max weight difference 0.0 (the dense matrix is ~99.7% zeros and the
eval VM is CPU-shared). The shipped `triage_model.npz` was **not** retrained —
Bitext's 27-intent schema differs from the product's 6 intents; the app and
its 13 tests are unchanged and passing.

### Priority: synthetic-only (no real labels)

Bitext has **no priority labels**, so priority was **not** evaluated on real
data. The priority numbers below remain **synthetic-only**.

### Preliminary results — synthetic data (`results.json`)

TF-IDF + multinomial logistic regression (NumPy, from scratch) on 600
**synthetic, template-generated** support messages written for this project
(480 train / 120 held-out test):

| Metric (synthetic, preliminary) | Result |
|---|---|
| Intent accuracy | 100% |
| Intent macro-F1 | 1.000 |
| Priority accuracy (**synthetic-only**) | 98.3% |
| Priority macro-F1 (**synthetic-only**) | 0.983 |
| Training time | ~14 s on CPU |

**Honest caveat:** these are an upper bound on easy, self-written templates —
the real-benchmark numbers above are the classifier claims. In delivery, the
classifier is re-tuned on each client's real ticket history (see BUSINESS.md).

## Architecture
```
web/index.html      landing + pricing          web/dashboard.html  owner UI
web/widget.html     embeddable chat widget  ──►  app.py  (http.server + sqlite3)
                                                    ├─ tenants / sessions / KB
                                                    ├─ retrieve()  TF-IDF per tenant
                                                    ├─ chat()      answer-or-ticket gate
                                                    └─ analytics() from DB rows
ml.py  TF-IDF + softmax LR (intent & priority heads) + sentiment + urgency rules
train_ml.py -> triage_model.npz + results.json
eval_bitext.py -> results_real.json  (real-benchmark eval; does not touch the shipped model)
```

## API reference
| Method & path | Auth | Purpose |
|---|---|---|
| POST `/api/register` | — | Create tenant → widget key |
| POST `/api/login` | — | Owner login → session token |
| GET `/api/me` | Bearer | Tenant info + widget key |
| GET/POST `/api/kb`, DELETE `/api/kb/<id>` | Bearer | Manage knowledge base |
| POST `/api/chat` | widget key in body | Customer chat → cited answer or ticket |
| GET `/api/tickets` | Bearer | Tickets, urgent-first |
| POST `/api/tickets/<id>/status` | Bearer | open / in-progress / resolved |
| GET `/api/analytics` | Bearer | Deflection, tickets, intents, confidence |
| GET `/api/demo-tenants` | — | Seeded tenants + widget keys (demo only) |

## Tech stack
Python 3.12 · stdlib http.server + sqlite3 · NumPy · vanilla HTML/CSS/JS · zero external services.

## Limitations & production path
- Demo-grade auth (no token expiry, no rate limiting, HTTP not HTTPS).
- Lexical TF-IDF retrieval — production swap: embeddings/LLM answers behind the
  same citation + confidence gate (roadmap in BUSINESS.md).
- SQLite suits MVP scale; Postgres at growth. No payments yet (Razorpay/Stripe planned).
- Triage numbers: real-benchmark intent results in the section above (Bitext,
  deduped split); priority figures are synthetic-only — no public dataset used
  here carries priority labels.

## Screenshots (described)
- **Landing:** hero "Your support inbox, answered by AI in seconds", 4 feature
  cards, 3 pricing tiers (₹4,999 / ₹9,999 / ₹19,999 + ₹14,999 setup).
- **Dashboard:** analytics stat cards, KB manager, priority-ordered ticket
  table, live chat tester showing raw triage JSON.
- **Widget demo:** fictional "UrbanKart" storefront with a working support
  chat bubble that cites articles and announces ticket creation.
