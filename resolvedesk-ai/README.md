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

## ML triage — real measured results (`results.json`)
TF-IDF + multinomial logistic regression (NumPy, from scratch) on 600 generated
support messages (480 train / 120 held-out test):

| Metric | Result |
|---|---|
| Intent accuracy | **100%** |
| Intent macro-F1 | **1.000** |
| Priority accuracy | **98.3%** |
| Priority macro-F1 | **0.983** |
| Training time | ~14 s on CPU |

**Honest caveat:** the dataset is synthetic and template-generated, so these
numbers are an upper bound — real customer language is messier. In delivery,
the classifier is re-tuned on each client's real ticket history (see BUSINESS.md).

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
- Triage numbers are on synthetic data (see caveat above).

## Screenshots (described)
- **Landing:** hero "Your support inbox, answered by AI in seconds", 4 feature
  cards, 3 pricing tiers (₹4,999 / ₹9,999 / ₹19,999 + ₹14,999 setup).
- **Dashboard:** analytics stat cards, KB manager, priority-ordered ticket
  table, live chat tester showing raw triage JSON.
- **Widget demo:** fictional "UrbanKart" storefront with a working support
  chat bubble that cites articles and announces ticket creation.
