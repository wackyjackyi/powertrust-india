# PowerTrust — India Solar Intelligence

**IE 7374 Generative AI Hackathon** | Open-Data Intelligence for Distributed Solar Development

A 5-layer RAG pipeline that ingests public regulatory documents, embeds them into a vector store, and serves a Streamlit dashboard with grounded AI chat, feasibility scoring, and unknown risk discovery — for solar development across **Rajasthan, Gujarat, and Karnataka**.

---

## Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://powertrust-india.streamlit.app/)

---

## Architecture

```
Layer 1 — Data Ingestion      layer1_ingest.py
  41 sources attempted → 23 ingested (6.7M chars)
  Sources: RERC, GERC, KERC, CEA, CERC, MNRE, IRENA, JMK

Layer 2 — Chunking + Embedding  layer2_embed.py
  3,252 chunks → FAISS index (4.8MB)
  Model: all-MiniLM-L6-v2 (sentence-transformers)

Layer 3 — RAG + LLM             app.py (load_rag, ask_question)
  ChromaDB in-memory + Groq (llama-3.3-70b-versatile)
  Multi-turn conversation with grounded context

Layer 4 — Streamlit UI          app.py
  5 pages: Visual Report, Chat, Insights & Scoring,
           Scalability, Data Audit

Layer 5 — Scoring + Insights    insights_dashboard.py
  6-dimension composite score (0-100)
  5 unknown unknowns discovery findings
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/powertrust-india.git
cd powertrust-india
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your Groq API key

Create a `.env` file or set the environment variable:

```bash
export GROQ_API_KEY=your_groq_api_key_here
```

Or for local Streamlit, create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

### 4. Run data ingestion (optional — pre-ingested files included)

```bash
# Ingest all sources (~3-5 min)
python layer1_ingest.py

# Test mode — first 3 sources only
python layer1_ingest.py --test

# Build FAISS index and ChromaDB chunks
python layer2_embed.py
```

### 5. Run the app

```bash
streamlit run app.py
```

---

## Deploy to Streamlit Cloud

1. Push this repo to GitHub (make sure `data/raw/*.txt` files are included)
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select your repo and set `app.py` as the entry point
4. Under **Secrets**, add:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
5. Deploy

> **Note:** Streamlit Cloud free tier has a 1GB memory limit. The embedding model (~90MB) and FAISS index (4.8MB) load at startup — this fits within limits.

---

## File Structure

```
powertrust-india/
├── app.py                    # Main Streamlit app (RAG + all pages)
├── insights_dashboard.py     # Scoring layer + unknown unknowns
├── layer1_ingest.py          # Data ingestion pipeline
├── layer2_embed.py           # Chunking + FAISS embedding
├── requirements.txt
├── README.md
└── data/
    ├── raw/                  # Ingested source documents (.txt)
    │   ├── rerc_tariff_rajasthan_full.txt
    │   ├── kerc_tariff_karnataka_full.txt
    │   ├── gerc_tariff_gujarat.txt
    │   ├── cea_national_electricity_plan.txt
    │   ├── cerc_rpo_regulations.txt
    │   ├── mnre_pm_surya_ghar.txt
    │   ├── jmk_solar_report_2025.txt
    │   ├── renewable_power_generation_full.txt
    │   └── ... (23 files total)
    ├── solar_india.index     # FAISS vector index
    ├── chunks_meta.pkl       # Chunk metadata
    ├── structured_data.json  # Regex-extracted structured values
    ├── state_data.csv        # Manually verified state metrics
    └── metadata.json         # Data audit report
```

---

## Data Sources

| Source | Type | State/Scope |
|---|---|---|
| RERC Discoms ARR & Tariff FY2025-26 | PDF | Rajasthan |
| GERC MYT Tariff Order 2025 | PDF | Gujarat |
| KERC Combined Tariff Order 2025 | PDF | Karnataka |
| CEA National Electricity Plan 2022-32 | PDF | National |
| CEA All India Installed Capacity Sep 2025 | PDF | National |
| CERC RPO Regulations 2022 | PDF | National |
| MNRE PM Surya Ghar Guidelines 2024 | PDF | National |
| IRENA Renewable Power Generation Costs 2023 | PDF | Global |
| JMK Solar Report FY2025 | HTML | India Market |

**19 additional sources attempted** — failures documented in `data/metadata.json` with impact assessment.

---

## App Pages

### 📊 Visual Report
10 charts sourced from ingested documents:
- RE capacity by state (CEA)
- Gujarat RPO trajectory 2025-2030 (GERC)
- Solar tariff comparison (RERC, GERC)
- India solar CAPEX trend (IRENA, MNRE)
- Global LCOE trend 2010-2023 (IRENA)
- Rajasthan RPO breakdown (RERC)
- PM Surya Ghar subsidy structure (MNRE)
- State comparison table
- National RE mix (CEA)
- India vs Global CAPEX (IRENA, CERC)

### 💬 Chat
RAG-powered Q&A grounded in ingested documents. Supports:
- Dimension queries
- Comparison queries
- Risk & gap queries
- Multi-turn follow-ups

### 📈 Insights & Scoring
- 6-dimension composite feasibility score (0-100) per state
- Radar chart comparison
- 5 unknown unknowns — country-specific risks not in the project brief

### 🌍 Scalability
How to extend PowerTrust to new countries (Malaysia, Mexico, Brazil planned).

### 📋 Data Audit
Full audit of found vs missing data with gap impact assessment.

---

## Scoring Methodology

| Dimension | Weight | Rajasthan | Gujarat | Karnataka |
|---|---|---|---|---|
| Cost competitiveness | 20% | 95 | 90 | 56 |
| Grid readiness | 20% | 35 | 65 | 65 |
| Policy environment | 20% | 80 | 95 | 60 |
| Land availability | 15% | 82 | 64 | 46 |
| Water risk | 10% | 20 | 45 | 70 |
| Approval speed | 15% | 64 | 88 | 76 |
| **Composite** | | **65.9** | **77.3** | **61.5** |

**Gujarat leads** on policy stability and rooftop solar penetration.

---

## Unknown Unknowns

Five non-obvious risks discovered from the corpus — not listed in the project brief:

1. **[Rajasthan] Panel Cleaning Water Crisis** — High severity
2. **[Rajasthan] Grid Evacuation Bottleneck — Bikaner-Khetri Corridor** — Critical severity
3. **[Gujarat] DISCOM Financial Health Undermines PPA Security** — Medium severity
4. **[Karnataka] Pavagada Farmer Lease Model Rate Dispute Risk** — High severity
5. **[Karnataka] Wind-Solar Curtailment in Southern Grid** — High severity

---

## Team

Built for IE 7374 Generative AI — Northeastern University
