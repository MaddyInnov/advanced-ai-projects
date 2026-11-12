# ResolveDesk AI — Business & Freelance Service Pack

*Owner/seller: Ashutosh Sahoo. This document is the sell-it-and-deliver-it playbook.*

## What you're selling
An AI support assistant installed on the client's own website: it answers customer
questions from the client's FAQs/policies, creates and prioritises tickets when it
can't answer, and gives the owner a dashboard showing how much support was deflected.
Sold as a **done-for-you service + subscription**, not as raw software.

## Target customers
- D2C / e-commerce stores (Shopify/WhatsApp sellers) drowning in "where is my order / refund?" messages
- Local service businesses: gyms, clinics, coaching institutes, salons, repair shops
- Small SaaS / ed-tech teams with repetitive support queries
Sweet spot: 200–5,000 support conversations/month, no dedicated support team.

## Pricing
| Tier | Price | Includes |
|---|---|---|
| Starter | ₹4,999/mo (~$59) | 1 business, 500 conversations/mo, KB chat widget, ticket triage |
| Growth | ₹9,999/mo (~$119) | 2,000 conversations/mo, analytics, monthly KB tuning, priority support |
| Pro | ₹19,999/mo (~$239) | Unlimited conversations, multiple brands, custom integrations |
| One-time setup | ₹14,999 (~$179) | KB loading, widget branding + install, triage tuning on client's real tickets, staff training call |

## Unit economics (ESTIMATE — labelled, not measured)
- Hosting per client (small VPS, shared): ~₹800–1,500/mo
- Maintenance/tuning time per client: ~2 hrs/mo
- At Growth tier: ~₹9,999 revenue − ~₹1,200 infra ≈ **~₹8,800/mo gross per client before time**
- 8 Growth clients ≈ ₹70k/mo recurring — in line with the owner's ₹60k+/mo target.
Estimates assume the current zero-API-cost stack; adding a paid LLM backend adds ~₹1–3 per conversation in API cost and must be priced into Pro.

## 48-hour client onboarding playbook (freelance delivery)
**Day 0 (call, 30 min):** collect FAQs, policies, price lists, past support chats (WhatsApp/email exports); agree escalation rules.
**Hours 0–6:** create tenant, load KB articles, paste the widget snippet into the client's site (or hand to their developer), brand colours/logo.
**Hours 6–24:** tune — run the client's 50 most common real questions through the tester; add/adjust articles until deflection on that set is satisfactory; verify urgent messages escalate.
**Hours 24–48:** owner training call (dashboard, tickets, adding articles), go live, hand over a one-page SOP. First-week check-in scheduled.

## One-page sales pitch
> **Your customers ask the same 20 questions every day. ResolveDesk AI answers them in seconds — and only bothers you with the rest, already sorted by urgency.**
> Installed on your website in 48 hours, trained on *your* policies (not generic AI guesses — when it doesn't know, it creates a ticket instead of making things up). You get a dashboard proving how many conversations it resolved without you. ₹4,999/month, set up for you. First demo runs on your own FAQs — if it can't answer them, you don't pay the setup fee.

## Competitor comparison (honest)
| | Intercom Fin / Zendesk AI | ResolveDesk AI |
|---|---|---|
| Pricing | Per-seat + per-resolution, USD, adds up fast for SMBs | Flat ₹/mo, SMB-first |
| Setup | Self-serve, client configures everything | Done-for-you in 48 hrs |
| Best for | Larger teams with support staff | Owner-run businesses with no support team |
| Where they win | Deeper integrations, mature LLM answers, enterprise security | — we don't claim parity; roadmap below closes the answer-quality gap |

## Roadmap to production
1. LLM answer layer (Groq/OpenRouter) behind the same citation + confidence gate, with the current lexical engine as fallback
2. WhatsApp channel (Meta Cloud API) — the channel Indian SMB customers actually use
3. Real auth (OTP/OAuth), rate limiting, HTTPS deployment, backups
4. Payments (Razorpay/Stripe) + plan enforcement by conversation quota
5. Train triage on each client's real ticket history; human-in-the-loop review queue
6. Hindi/Hinglish KB support
