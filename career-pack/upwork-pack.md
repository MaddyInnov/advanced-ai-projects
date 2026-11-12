# Upwork Pack — Ashutosh Sahoo
### Profile copy, service offerings and proposal templates (November 2026)

All results quoted below come from the six verified projects in this portfolio. Each
offering names its demonstration project so a client can inspect the code and report.
Prices are starting points for a first set of reviews; raise after 3–5 completed jobs.

---

## Profile Title Options (pick one, rotate monthly)

1. **AI Engineer | LLM Fine-Tuning (LoRA/QLoRA), RAG Chatbots & Custom ML Models**
2. **Generative AI Developer — Fine-Tuned LLMs, Agentic RAG & AI Support Systems for SMBs**
3. **ML/DL Engineer (M.Tech, NIT) — PyTorch, LLM Fine-Tuning, RAG, Model Evaluation**

## Profile Overview (~250 words)

I am an AI Engineer with an M.Tech in Artificial Intelligence from NIT Silchar (CGPA 8.82), and I build AI systems end-to-end — model, evaluation, backend and a demo you can click.

My recent work, all open to inspection with full written reports:

• **LLM fine-tuning:** I implemented LoRA from scratch in PyTorch and matched full fine-tuning accuracy while training only 1.86% of the parameters, with a documented hyperparameter sweep and a ready PEFT/QLoRA pipeline for models such as Qwen and Llama.
• **RAG & support AI:** I built an agentic RAG system with hybrid retrieval (TF-IDF + BM25 + reranking), grounded and cited answers, correct abstention when the documents do not contain the answer, and prompt-injection guardrails — every claim measured by an evaluation harness, not asserted. I also productised this into ResolveDesk AI, a working multi-tenant AI support desk for small businesses.
• **Custom ML/DL:** from-scratch diffusion models (DDPM/DDIM), temporal segmentation research extending my published-style InceptionKAN work (97.25% on UCI-HAR), and federated learning with differential privacy.

How I work: you get a working demo early, an evaluation report with baselines and limitations stated plainly, clean documented code, and a short handover walkthrough. I do not quote accuracy numbers I have not measured, and I will tell you before we start if your data is not ready for the result you want.

Immediate availability, India-based, working with clients globally. Message me with your use case and a sample of your data — I will reply with a concrete plan and a fixed quote before any contract starts.

---

## Service Offering 1 — LLM Fine-Tuning (LoRA/QLoRA) for Your Domain

**What you get:** a model fine-tuned on *your* documents, tickets, chats or records, so it
answers in your domain's language instead of generic LLM output.

**Demonstration project:** Sentinel-LLM — LoRA implemented from scratch; fine-tuned model
matched full fine-tuning at 1.86% trainable parameters across 6 cybersecurity tasks, with a
documented rank/learning-rate sweep and per-task evaluation.

**Deliverables**
- Data audit + instruction-dataset built from your raw material (cleaning, formatting, train/val/test splits)
- LoRA/QLoRA fine-tune of an agreed open model (e.g. Qwen / Llama family) on GPU
- Evaluation report: base model vs fine-tuned model on a held-out set, per-task/per-category breakdown, failure examples, and limitations stated plainly
- Inference package: adapter weights, a runnable demo/API, and a README your team can operate
- One revision round on data or hyperparameters

**Starting price:** from **$299 / ₹24,999** for a first fine-tune (single dataset up to ~2,000 examples, one model, standard evaluation). Larger datasets, multiple models or deployment support quoted after a free data review.

## Service Offering 2 — RAG Chatbot / AI Support Desk Setup (ResolveDesk, productised)

**What you get:** an AI assistant on your website that answers customers from *your* FAQs
and policies — and creates a support ticket instead of inventing an answer when it does not know.

**Live product:** ResolveDesk AI — multi-tenant support SaaS built end-to-end (knowledge
base, cited chat widget, ML ticket triage at 100% intent / 98.3% priority accuracy on its
held-out test set, owner dashboard with deflection analytics; 13/13 tests, 10/10 end-to-end
smoke test).

**Deliverables**
- Your knowledge base loaded and structured (FAQs, policies, manuals, help articles)
- Branded chat widget installed on your site, answering with citations
- Automatic ticket creation + intent/priority triage for questions the AI cannot answer
- Owner dashboard: conversations, AI-resolved (deflection) rate, ticket queue
- Triage classifier tuned on your real past tickets, plus a short staff training call
- 48-hour standard setup; monthly plans include knowledge-base tuning

**Pricing:** one-time setup **₹14,999 (~$179)** · then **Starter ₹4,999/mo (~$59)** ·
**Growth ₹9,999/mo (~$119)** · **Pro ₹19,999/mo (~$239)**. Custom RAG builds (internal
documents, other channels such as WhatsApp) quoted per scope, from **$499**.

## Service Offering 3 — Custom AI/ML Model Development & Evaluation

**What you get:** a model built — or an existing model honestly evaluated — for your
specific problem: classification, time-series/activity recognition, anomaly triage,
recommendation features, or a research prototype.

**Demonstration projects:** DenseHAR (dense temporal segmentation, subject-disjoint
evaluation), DiffusionForge (diffusion models from scratch), FedGuard (federated learning
with differential privacy) — each shipped with baselines, ablations and a thesis-style report.

**Deliverables**
- Problem framing + data audit with a written feasibility note before modelling starts
- Baseline first, then the proposed model — both evaluated on the same held-out protocol
- Ablations for any claimed improvement, and segmental/task-specific metrics where accuracy alone misleads
- Clean PyTorch code, trained weights, a runnable demo, and a written report a non-author can learn from
- Handover call + one revision round

**Starting price:** from **$249 / ₹19,999** for a scoped prototype or evaluation study.
Research-grade or multi-model work quoted per milestone.

---

## Proposal Template A — LLM Fine-Tuning Gig

> Subject: Fine-tuning [model family, e.g. Qwen/Llama] on your [data type] — plan + fixed quote
>
> Hi [Client name],
>
> I read your post about [their exact use case — e.g. "classifying support tickets into your 12 categories"]. I do this class of work regularly: my most recent fine-tuning project implemented LoRA from scratch and matched full fine-tuning while training only 1.86% of the parameters, evaluated per-task on a held-out test set — the code and full report are on my profile if you would like to inspect the depth before we talk.
>
> For your data, my plan would be:
> 1. **Data first:** audit your [N] examples, clean and format them as instructions, and hold out a test set we both trust before any training.
> 2. **Baseline:** measure the base model (and a prompt-only approach) on that test set, so the fine-tune has to *earn* its cost.
> 3. **Fine-tune:** LoRA/QLoRA on [agreed model], with a small rank/learning-rate sweep selected on validation — not on the test set.
> 4. **Deliver:** adapter weights, a runnable demo, and an evaluation report comparing base vs fine-tuned per category, including failure examples.
>
> Two questions so I can quote exactly: (1) Can you share 20–30 sample rows (sensitive fields removed)? (2) Is [their stated success metric] the number that decides success for you?
>
> Timeline: [X days] from data handover. Fixed price: [$ amount], milestone-split as [data+baseline / fine-tune+report].
>
> — Ashutosh Sahoo, M.Tech (AI), NIT Silchar · github.com/Ashutosh130801

## Proposal Template B — RAG Chatbot / AI Support Desk Gig

> Subject: AI support assistant on your [website/help centre] — working demo first
>
> Hi [Client name],
>
> Your post on [their exact problem — e.g. "answering the same delivery and refund questions all day"] is exactly what I productised in ResolveDesk AI, a support desk I built end-to-end: customers get cited answers from your own FAQs, and when the AI does not know, it creates a triaged ticket instead of guessing. It is live code with tests and an evaluation report — happy to share the demo link.
>
> What I would set up for [their business name]:
> 1. Load your [FAQs / help articles / policy pages — name their actual sources] into the knowledge base (48-hour standard setup).
> 2. Install the branded chat widget on [their platform — e.g. WordPress/Shopify/custom site].
> 3. Tune the ticket triage (intent + priority) on your past [tickets/emails], and set the confidence gate so uncertain questions reach your team, not your customers.
> 4. Hand over the owner dashboard — conversations, AI-resolved rate, open tickets — with a short training call for [team size] staff.
>
> Pricing is transparent: ₹14,999 (~$179) one-time setup, then ₹[4,999/9,999]/month depending on conversation volume — or a fixed [$ amount] for a custom build if you prefer no subscription.
>
> If you send me your top 10 customer questions, I will run them through the demo on your own FAQs before you pay anything — if it cannot answer them well, I will tell you why and what it would take.
>
> — Ashutosh Sahoo, M.Tech (AI), NIT Silchar · github.com/Ashutosh130801

---

*Personalisation rule: never send a template with a bracket left in. Every bracket maps to
something specific in the client's post — that specificity is the proposal.*
