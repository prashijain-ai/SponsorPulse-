<div align="center">

# 🚀 SponsorPulse: TabPFN-3.5 + Gemma 3:1b Local AI Negotiation Agent
### *The Ultimate Open-Source Side Hustle Tool for Creators, UGC Artists, & Indie Makers*

[![Hacktoberfest 2026](https://img.shields.io/badge/Hacktoberfest-2026-orange?style=for-the-badge&logo=hacktoberfest)](https://dev.to)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![TabPFN](https://img.shields.io/badge/TabPFN-3.5%20Prior%20Labs-purple?style=for-the-badge)](https://priorlabs.ai)
[![Google Gemma](https://img.shields.io/badge/Google-Gemma%203:1b-pink?style=for-the-badge&logo=google)](https://ai.google.dev/)
[![Sentry](https://img.shields.io/badge/Sentry-Agent%20Tracing-orange?style=for-the-badge&logo=sentry)](https://sentry.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/mit)

<p align="center">
  <b>Never leave money on the table again. SponsorPulse runs a rigorous TabPFN tabular prediction engine alongside local Gemma 3:1b to audit brand lowballs, scan predatory contracts, and generate bulletproof counter-offer scripts—100% locally on Apple Silicon.</b>
</p>

</div>

---

## 💡 The Problem & Open Innovation Rationale

Every week, thousands of micro-influencers, newsletter writers, and UGC creators receive brand sponsorship offers containing **predatory contract clauses** (like uncompensated 90-day whitelisting, 30-day competitor lockouts, and severe budget lowballs). 

Creator financial data (past brand payouts, unreleased campaign briefs, private NDAs) is extremely sensitive. Creators cannot upload this data to closed cloud LLMs without violating confidentiality or paying expensive monthly SaaS fees. 

**SponsorPulse solves this via Strict Separation Architecture:**
1. **Zero LLM Math:** TabPFN computes calibrated 10th, 50th, and 90th percentile valuation bands and runs a 5x parallel **Counterfactual What-If Agent** loop.
2. **Local Gemma 3:1b Enforcement:** Google's `gemma3:1b` model runs locally via Ollama (lightning-fast on M4 Macs at 120+ tokens/sec) to weave TabPFN's exact quantiles into high-converting negotiation emails.
3. **Deterministic Numerical Firewall:** An AST/Regex verification layer inspects Gemma's output in real time to physically block hallucinated numbers.

---

## 🛠️ Architecture & Tech Stack

* **Predictive Core:** `TabPFN-3.5` / Scikit-Learn Gradient Boosting Quantile Regressors.
* **Local Language Agent:** Google `gemma3:1b` hosted locally via `Ollama`.
* **Backend:** Python FastAPI with async multi-query simulation loops.
* **Frontend:** Tailwind CSS, Chart.js probability curves, and an embedded **1-Click Judge Sandbox** (works both locally via FastAPI and statically on GitHub Pages!).
* **Observability:** Sentry SDK with custom OpenTelemetry span instrumentation.

---

## ⚡ Quickstart Guide (Run Locally in 60 Seconds)

Ensure you have Python 3.10+ and [Ollama](https://ollama.com/) installed on your machine.

```bash
# 1. Clone the repository
git clone [https://github.com/YOUR_USERNAME/sponsorpulse.git](https://github.com/YOUR_USERNAME/sponsorpulse.git)
cd sponsorpulse

# 2. Pull local Gemma 3:1b model
ollama pull gemma3:1b

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch FastAPI server
uvicorn app:app --reload --port 8000