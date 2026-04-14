"""
PowerTrust India — Layer 1: Data Ingestion
==========================================
Covers all 6 required dimensions across Rajasthan + Gujarat + Karnataka (+ national sources).
Run this script to build the raw corpus before embedding (Layer 2/3).

Usage:
    python layer1_ingest.py                  # ingest all sources
    python layer1_ingest.py --dim costs      # ingest one dimension only
    python layer1_ingest.py --test           # dry run, prints URLs only

Output:
    data/raw/          raw text files, one per source
    data/metadata.json summary of what was found / missing
"""

import os
import io
import json
import time
import argparse
import hashlib
import logging
from datetime import date
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import urllib3
import requests
from bs4 import BeautifulSoup
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)



try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False
    print("[WARN] pypdf not installed — PDF ingestion disabled. Run: pip install pypdf")

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Output dirs ───────────────────────────────────────────────────────────────
RAW_DIR  = Path("data/raw")
META_DIR = Path("data")
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

TODAY = str(date.today())
DELAY = 1.5  # seconds between requests — be polite to government servers


# ── Data model ────────────────────────────────────────────────────────────────
@dataclass
class SourceRecord:
    """One ingested document."""
    source_id:      str            # e.g. "mnre_benchmark_capex"
    source_name:    str            # human label
    url:            str
    dimension:      str            # one of the 6 rubric dimensions
    state_scope:    str            # "national" | "rajasthan" | "gujarat" | "karnataka" | "all_three"
    format:         str            # "pdf" | "html" | "csv"
    status:         str = "pending"   # "ok" | "failed" | "skipped"
    char_count:     int = 0
    date_accessed:  str = TODAY
    notes:          str = ""
    text_file:      str = ""       # path to saved raw text


# ── Source catalogue ──────────────────────────────────────────────────────────
# Add or remove entries here — the ingestion loop handles the rest.
# "url" can be:
#   - a direct PDF link         → extracted with pypdf
#   - an HTML page              → extracted with BeautifulSoup
#   - a CSV link                → saved as-is (handled in Layer 3)

SOURCES: list[SourceRecord] = [

    # ── DIMENSION 1: Cost & Economics ─────────────────────────────────────────
    SourceRecord(
        source_id   = "mnre_benchmark_cost_2024",
        source_name = "MNRE Benchmark Capital Cost 2024-25",
        url         = "https://mnre.gov.in/img/documents/uploads/file_f-1714034284700.pdf",
        dimension   = "cost_economics",
        state_scope = "national",
        format      = "pdf",
        notes       = "Annual benchmark CAPEX for solar PV — key for cost layer",
    ),
    SourceRecord(
        source_id   = "cerc_tariff_order_2024",
        source_name = "CERC Tariff Determination Order 2024",
        url         = "https://cercind.gov.in/2024/ORDER/sm.pdf",
        dimension   = "cost_economics",
        state_scope = "national",
        format      = "pdf",
        notes       = "National tariff benchmarks; compare with SECI auction results",
    ),
    SourceRecord(
        source_id   = "seci_auction_results",
        source_name = "SECI Auction Results — JMK Research summary",
        url         = "https://jmkresearch.com/indian-solar-market-update-fy2025/",
        dimension   = "cost_economics",
        state_scope = "national",
        format      = "html",
        notes       = "JMK Research summary of SECI auction tariffs — more scraper-friendly than SECI directly",
    ),
    SourceRecord(
        source_id   = "irena_india_costs",
        source_name = "IRENA India renewable cost data — Our World in Data",
        url         = "https://ourworldindata.org/grapher/levelized-cost-of-energy",
        dimension   = "cost_economics",
        state_scope = "national",
        format      = "html",
        notes       = "LCOE benchmarks in accessible format — cross-technology comparison",
    ),
    SourceRecord(
        source_id   = "rerc_tariff_rajasthan",
        source_name = "RERC Tariff Order Rajasthan 2023-24",
        url         = "https://www.rerc.rajasthan.gov.in/rerc-user-files/tariff-order/TO_FY2023-24_DISCOMS.pdf",
        dimension   = "cost_economics",
        state_scope = "rajasthan",
        format      = "pdf",
        notes       = "State-level retail tariff — needed for grid charge calculation",
    ),
    SourceRecord(
        source_id   = "gerc_tariff_gujarat",
        source_name = "GERC Tariff Order Gujarat 2023-24",
        url         = "https://www.gercin.org/wp-content/uploads/2023/05/TO-2023-24-English.pdf",
        dimension   = "cost_economics",
        state_scope = "gujarat",
        format      = "pdf",
        notes       = "Gujarat state tariff for cost comparison with Rajasthan",
    ),

    # ── DIMENSION 2: Grid Access & Queue Dynamics ──────────────────────────────
    SourceRecord(
        source_id   = "cea_national_electricity_plan",
        source_name = "CEA National Electricity Plan 2022-32",
        url         = "https://cea.nic.in/wp-content/uploads/irp/2023/05/NEP_2022_32_FINAL.pdf",
        dimension   = "grid_access",
        state_scope = "national",
        format      = "pdf",
        notes       = "Transmission expansion plans; look for Rajasthan/Gujarat chapters",
    ),
    SourceRecord(
        source_id   = "posoco_congestion_report",
        source_name = "India grid data — Ember Energy India",
        url         = "https://ember-energy.org/countries/india/",
        dimension   = "grid_access",
        state_scope = "national",
        format      = "html",
        notes       = "Ember India page — grid RE capacity, curtailment and congestion trends",
    ),
    SourceRecord(
        source_id   = "open_access_registry",
        source_name = "Open Access Registry India",
        url         = "https://openaccessregistry.com/",
        dimension   = "grid_access",
        state_scope = "all_three",
        format      = "html",
        notes       = "State-wise open access approvals; may need manual download",
    ),
    SourceRecord(
        source_id   = "rerc_open_access_rajasthan",
        source_name = "RERC Open Access Regulations",
        url         = "https://www.rerc.rajasthan.gov.in/rerc-user-files/regulations/Open_Access_Regulations_2019.pdf",
        dimension   = "grid_access",
        state_scope = "rajasthan",
        format      = "pdf",
        notes       = "Rajasthan open access rules — interconnection wait time info",
    ),
    SourceRecord(
        source_id   = "gerc_open_access_gujarat",
        source_name = "GERC Open Access Regulations",
        url         = "https://www.gercin.org/regulations/",
        dimension   = "grid_access",
        state_scope = "gujarat",
        format      = "html",
        notes       = "Scrape Gujarat open access regulation documents",
    ),

    # ── DIMENSION 3: Subsidies, Incentives & Policy ────────────────────────────
    SourceRecord(
        source_id   = "mnre_pm_surya_ghar",
        source_name = "PM Surya Ghar Muft Bijli Yojana Guidelines",
        url         = "https://mnre.gov.in/img/documents/uploads/file_f-1707121512931.pdf",
        dimension   = "subsidies_policy",
        state_scope = "national",
        format      = "pdf",
        notes       = "Key residential rooftop solar subsidy scheme — launched Feb 2024",
    ),
    SourceRecord(
        source_id   = "mnre_rooftop_phase2",
        source_name = "MNRE Rooftop Solar Programme Phase II",
        url         = "https://mnre.gov.in/solar/rooftop",
        dimension   = "subsidies_policy",
        state_scope = "national",
        format      = "html",
        notes       = "Programme overview + state-wise targets",
    ),
    SourceRecord(
        source_id   = "cea_net_metering_framework",
        source_name = "CEA Net Metering Framework",
        url         = "https://cea.nic.in/wp-content/uploads/notification/2022/06/CEA_Regulation_Connectivity.pdf",
        dimension   = "subsidies_policy",
        state_scope = "national",
        format      = "pdf",
        notes       = "Model net metering regulations — basis for state-level rules",
    ),
    SourceRecord(
        source_id   = "datagov_subsidy_statewise",
        source_name = "data.gov.in State-wise Rooftop Solar Incentives",
        url         = "https://www.data.gov.in/catalog/state-wise-incentive-amount-released-power-distribution-companies-discoms-under-phase-ii",
        dimension   = "subsidies_policy",
        state_scope = "all_three",
        format      = "html",
        notes       = "Open government dataset — state-wise subsidy disbursements",
    ),
    SourceRecord(
        source_id   = "niti_aayog_energy_policy",
        source_name = "NITI Aayog India Energy Security Scenarios",
        url         = "https://www.niti.gov.in/sites/default/files/2023-01/National_Energy_Policy_NITI.pdf",
        dimension   = "subsidies_policy",
        state_scope = "national",
        format      = "pdf",
        notes       = "Overarching subsidy and incentive policy framework",
    ),

    # ── DIMENSION 4: Utility Standards & Obligations ───────────────────────────
    SourceRecord(
        source_id   = "cerc_rpo_regulations",
        source_name = "CERC RPO Regulations 2022",
        url         = "https://cercind.gov.in/2022/regulation/noti290922.pdf",
        dimension   = "utility_standards",
        state_scope = "national",
        format      = "pdf",
        notes       = "Renewable Purchase Obligation targets — solar-specific sub-targets",
    ),
    SourceRecord(
        source_id   = "ministry_power_nep",
        source_name = "Ministry of Power National Electricity Policy",
        url         = "https://powermin.gov.in/sites/default/files/uploads/National_Electricity_Policy_2021.pdf",
        dimension   = "utility_standards",
        state_scope = "national",
        format      = "pdf",
        notes       = "Utility procurement obligations and RE integration mandates",
    ),
    SourceRecord(
        source_id   = "rerc_rpo_rajasthan",
        source_name = "RERC RPO Targets Rajasthan 2023-24",
        url         = "https://www.rerc.rajasthan.gov.in/rerc-user-files/orders/RPO_Order_2023.pdf",
        dimension   = "utility_standards",
        state_scope = "rajasthan",
        format      = "pdf",
        notes       = "State-specific RPO compliance orders",
    ),
    SourceRecord(
        source_id   = "gerc_rpo_gujarat",
        source_name = "GERC RPO Targets Gujarat",
        url         = "https://www.gercin.org/orders-judgments/",
        dimension   = "utility_standards",
        state_scope = "gujarat",
        format      = "html",
        notes       = "Scrape for latest RPO compliance orders",
    ),

    # ── DIMENSION 5: Public Comment & Approval Signals ────────────────────────
    SourceRecord(
        source_id   = "moef_eia_solar",
        source_name = "MoEFCC EIA Notifications — Solar Projects",
        url         = "https://moef.gov.in/moef/division/environment-impact-assessment-division/eia/",
        dimension   = "approval_signals",
        state_scope = "national",
        format      = "html",
        notes       = "Public hearing transcripts — unstructured but rich for NLP",
    ),
    SourceRecord(
        source_id   = "pib_solar_approvals",
        source_name = "PIB Press Releases — Solar Projects",
        url         = "https://pib.gov.in/allRel.aspx",
        dimension   = "approval_signals",
        state_scope = "national",
        format      = "html",
        notes       = "Search 'solar' — project announcements, approval notices",
    ),
    SourceRecord(
        source_id   = "prs_solar_policy",
        source_name = "PRS Legislative Research — Solar/RE Bills",
        url         = "https://prsindia.org/policy/energy",
        dimension   = "approval_signals",
        state_scope = "national",
        format      = "html",
        notes       = "Policy debate summaries, stakeholder objection themes",
    ),
    SourceRecord(
        source_id   = "ngt_solar_cases",
        source_name = "NGT Case Filings — Solar Land Disputes",
        url         = "https://greentribunal.gov.in/",
        dimension   = "approval_signals",
        state_scope = "national",
        format      = "html",
        notes       = "Land-use objections, environmental challenge patterns",
    ),

    # ── KARNATAKA STATE SOURCES (all dimensions) ──────────────────────────────
    SourceRecord(
        source_id   = "kerc_tariff_karnataka",
        source_name = "KERC Tariff Order Karnataka 2023-24",
        url         = "https://kerc.karnataka.gov.in/uploads/tariff_orders/1712207605.pdf",
        dimension   = "cost_economics",
        state_scope = "karnataka",
        format      = "pdf",
        notes       = "Karnataka retail tariff — compare with Rajasthan & Gujarat for cost layer",
    ),
    SourceRecord(
        source_id   = "kerc_solar_procurement",
        source_name = "KERC Solar Procurement Orders",
        url         = "https://kerc.karnataka.gov.in/page/orders+and+judgements/tariff+orders/index.html",
        dimension   = "cost_economics",
        state_scope = "karnataka",
        format      = "html",
        notes       = "Scrape for solar PPA tariff orders — Karnataka discovered tariffs",
    ),
    SourceRecord(
        source_id   = "kerc_open_access_karnataka",
        source_name = "KERC Open Access Regulations Karnataka",
        url         = "https://kerc.karnataka.gov.in/page/regulations/index.html",
        dimension   = "grid_access",
        state_scope = "karnataka",
        format      = "html",
        notes       = "Open access + grid interconnection rules — Pavagada Solar Park context",
    ),
    SourceRecord(
        source_id   = "bescom_grid_karnataka",
        source_name = "BESCOM Annual Report / Grid Data",
        url         = "https://bescom.karnataka.gov.in/page/Report/Annual+Reports/en",
        dimension   = "grid_access",
        state_scope = "karnataka",
        format      = "html",
        notes       = "BESCOM (Bangalore utility) — grid congestion and interconnection wait data",
    ),
    SourceRecord(
        source_id   = "kerc_net_metering_karnataka",
        source_name = "KERC Net Metering Regulations Karnataka",
        url         = "https://kerc.karnataka.gov.in/uploads/regulations/1608536079.pdf",
        dimension   = "subsidies_policy",
        state_scope = "karnataka",
        format      = "pdf",
        notes       = "Karnataka net metering rules — compare NEM policies across 3 states",
    ),
    SourceRecord(
        source_id   = "kredl_karnataka",
        source_name = "KREDL Karnataka RE Policy",
        url         = "https://kredl.karnataka.gov.in/page/New+Energy+Policy/index.html",
        dimension   = "subsidies_policy",
        state_scope = "karnataka",
        format      = "html",
        notes       = "Karnataka Renewable Energy Development Ltd — state subsidy schemes",
    ),
    SourceRecord(
        source_id   = "kerc_rpo_karnataka",
        source_name = "KERC RPO Targets Karnataka",
        url         = "https://kerc.karnataka.gov.in/page/orders+and+judgements/rpo+orders/index.html",
        dimension   = "utility_standards",
        state_scope = "karnataka",
        format      = "html",
        notes       = "State RPO compliance orders — Karnataka has one of highest RE shares (69%)",
    ),
    SourceRecord(
        source_id   = "pavagada_park_data",
        source_name = "Pavagada Solar Park — project records",
        url         = "https://pib.gov.in/newsite/PrintRelease.aspx?relid=177234",
        dimension   = "approval_signals",
        state_scope = "karnataka",
        format      = "html",
        notes       = "PIB release on Pavagada — land acquisition, farmer lease model detail",
    ),
    SourceRecord(
        source_id   = "karnataka_solar_capacity",
        source_name = "Karnataka state solar capacity stats",
        url         = "https://mnre.gov.in/img/documents/uploads/file_f-1712126209688.pdf",
        dimension   = "unknown_unknowns",
        state_scope = "karnataka",
        format      = "pdf",
        notes       = "MNRE state-wise capacity — Karnataka ~9.9 GW solar, wind+solar mix insight",
    ),

    # ── DIMENSION 6: Unknown Unknowns (Discovery Layer) ───────────────────────
    SourceRecord(
        source_id   = "prayas_re_portal",
        source_name = "Prayas Energy Group RE Data Portal",
        url         = "https://energy.prayaspune.org/renewable-energy-data-portal",
        dimension   = "unknown_unknowns",
        state_scope = "all_three",
        format      = "html",
        notes       = "Pre-aggregated state RE stats — good cross-check source",
    ),
    SourceRecord(
        source_id   = "jmk_solar_report_2025",
        source_name = "JMK Research FY2025 Solar Report",
        url         = "https://jmkresearch.com/4-2-gw-wind-and-23-8-gw-solar-installed-in-india-in-fy2025/",
        dimension   = "unknown_unknowns",
        state_scope = "all_three",
        format      = "html",
        notes       = "State-wise solar additions FY2025 — good for unknown bottlenecks",
    ),
    SourceRecord(
        source_id   = "cea_installed_capacity",
        source_name = "CEA All India Installed Capacity (monthly)",
        url         = "https://cea.nic.in/installed-capacity-report/",
        dimension   = "unknown_unknowns",
        state_scope = "all_three",
        format      = "html",
        notes       = "Latest state-wise capacity data — anchor for dashboard charts",
    ),
    SourceRecord(
        source_id   = "datagov_solar_capacity",
        source_name = "data.gov.in State-wise Solar Capacity",
        url         = "https://www.data.gov.in/catalog/state-wise-solar-power-capacity-installed-india",
        dimension   = "unknown_unknowns",
        state_scope = "all_three",
        format      = "html",
        notes       = "Machine-readable capacity data — use for dashboard charts",
    ),

    # ── TEAMMATE-ADDED SOURCES ─────────────────────────────────────────────────
    SourceRecord(
        source_id   = "renewable_power_generation_full",
        source_name = "IRENA Renewable Power Generation Costs 2023",
        url         = "https://www.irena.org/Publications/2024/Sep/Renewable-Power-Generation-Costs-in-2023",
        dimension   = "cost_economics",
        state_scope = "national",
        format      = "pdf",
        notes       = "Global LCOE benchmarks — solar PV USD 758/kW, USD 0.044/kWh in 2023. Key for cross-technology comparison.",
    ),
    SourceRecord(
        source_id   = "cea_installed_capacity_full",
        source_name = "CEA All India Installed Capacity — full report",
        url         = "https://cea.nic.in/installed-capacity-report/",
        dimension   = "unknown_unknowns",
        state_scope = "all_three",
        format      = "pdf",
        notes       = "Full state-wise installed capacity data from CEA — anchor for dashboard charts",
    ),
    SourceRecord(
        source_id   = "kerc_tariff_karnataka_full",
        source_name = "KERC Combined Tariff Order Karnataka 2025 — full text",
        url         = "https://data.opencity.in/dataset/kerc-combined-tariff-order-2025",
        dimension   = "cost_economics",
        state_scope = "karnataka",
        format      = "pdf",
        notes       = "Full extracted text — Karnataka tariff, RPO, wheeling charges across all ESCOMs",
    ),
    SourceRecord(
        source_id   = "rerc_tariff_rajasthan_full",
        source_name = "RERC Discoms ARR and Tariff FY2025-26 — full text",
        url         = "https://rerc.rajasthan.gov.in/rerc-user-files/tariff-orders",
        dimension   = "cost_economics",
        state_scope = "rajasthan",
        format      = "pdf",
        notes       = "Full extracted text — Rajasthan tariff order, RPO, solar procurement data",
    ),
]


# ── Ingestion helpers ─────────────────────────────────────────────────────────

def _save_text(source_id: str, text: str) -> str:
    """Write raw text to data/raw/<source_id>.txt, return filepath."""
    path = RAW_DIR / f"{source_id}.txt"
    path.write_text(text, encoding="utf-8")
    return str(path)


def ingest_pdf(rec: SourceRecord) -> SourceRecord:
    """Download a PDF and extract all text."""
    if not HAS_PYPDF:
        rec.status = "skipped"
        rec.notes += " [pypdf not installed]"
        return rec
    try:
        log.info(f"  PDF  {rec.source_id}")
        r = requests.get(rec.url, headers=HEADERS, timeout=45, verify=False)
        r.raise_for_status()
        reader = pypdf.PdfReader(io.BytesIO(r.content))
        pages = [p.extract_text() or "" for p in reader.pages]
        text = "\n\n".join(pages)
        if len(text.strip()) < 100:
            rec.status = "failed"
            rec.notes += " [PDF extracted <100 chars — may be scanned/image PDF]"
            return rec
        rec.text_file = _save_text(rec.source_id, text)
        rec.char_count = len(text)
        rec.status = "ok"
    except requests.exceptions.HTTPError as e:
        rec.status = "failed"
        rec.notes += f" [HTTP {e.response.status_code}]"
    except Exception as e:
        rec.status = "failed"
        rec.notes += f" [{type(e).__name__}: {e}]"
    return rec


def ingest_html(rec: SourceRecord) -> SourceRecord:
    """Fetch an HTML page and extract readable text."""
    try:
        log.info(f"  HTML {rec.source_id}")
        r = requests.get(rec.url, headers=HEADERS, timeout=30, verify=False)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        # Remove nav, footer, scripts
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        # Extract meaningful text blocks
        blocks = soup.find_all(["p", "h1", "h2", "h3", "h4", "li", "td", "th"])
        text = "\n".join(b.get_text(strip=True) for b in blocks if b.get_text(strip=True))
        if len(text.strip()) < 100:
            rec.status = "failed"
            rec.notes += " [Extracted <100 chars — page may require JS]"
            return rec
        rec.text_file = _save_text(rec.source_id, text)
        rec.char_count = len(text)
        rec.status = "ok"
    except requests.exceptions.HTTPError as e:
        rec.status = "failed"
        rec.notes += f" [HTTP {e.response.status_code}]"
    except Exception as e:
        rec.status = "failed"
        rec.notes += f" [{type(e).__name__}: {e}]"
    return rec


def ingest_manual(rec: SourceRecord) -> SourceRecord:
    """
    Load a file that was manually downloaded and saved to data/raw/<source_id>.txt
    or data/raw/<source_id>.pdf.
    Use this for sites that block automated scraping (403, connection reset).
    """
    txt_path = RAW_DIR / f"{rec.source_id}.txt"
    pdf_path = RAW_DIR / f"{rec.source_id}.pdf"

    if txt_path.exists():
        text = txt_path.read_text(encoding="utf-8", errors="ignore")
        rec.char_count = len(text)
        rec.text_file  = str(txt_path)
        rec.status     = "ok"
        rec.notes     += " [manually loaded from .txt]"
        return rec

    if pdf_path.exists() and HAS_PYPDF:
        reader = pypdf.PdfReader(str(pdf_path))
        text   = "\n\n".join(p.extract_text() or "" for p in reader.pages)
        rec.text_file  = _save_text(rec.source_id, text)
        rec.char_count = len(text)
        rec.status     = "ok"
        rec.notes     += " [manually loaded from .pdf]"
        return rec

    rec.status  = "failed"
    rec.notes  += " [manual file not found — save PDF/text to data/raw/]"
    return rec


def ingest_source(rec: SourceRecord, dry_run: bool = False) -> SourceRecord:
    """Route to the correct ingestion function."""
    if dry_run:
        log.info(f"  DRY  [{rec.dimension}] {rec.source_id}  →  {rec.url}")
        rec.status = "skipped"
        return rec
    # Check if manually downloaded file exists first
    if (RAW_DIR / f"{rec.source_id}.txt").exists() or \
       (RAW_DIR / f"{rec.source_id}.pdf").exists():
        log.info(f"  MAN  {rec.source_id} (using local file)")
        return ingest_manual(rec)
    if rec.format == "pdf":
        return ingest_pdf(rec)
    elif rec.format in ("html", "csv"):
        return ingest_html(rec)
    else:
        rec.status = "skipped"
        rec.notes += f" [unknown format: {rec.format}]"
        return rec


# ── Metadata writer ───────────────────────────────────────────────────────────

def write_metadata(results: list[SourceRecord]):
    """
    Write data/metadata.json — used by the data audit tab in Streamlit.
    Tracks what was found, what failed, and gap impact.
    """
    ok      = [r for r in results if r.status == "ok"]
    failed  = [r for r in results if r.status == "failed"]
    skipped = [r for r in results if r.status == "skipped"]

    # Group gaps by dimension
    gap_by_dim: dict[str, list[str]] = {}
    for r in failed:
        gap_by_dim.setdefault(r.dimension, []).append(r.source_id)

    meta = {
        "generated":   TODAY,
        "total":       len(results),
        "ok":          len(ok),
        "failed":      len(failed),
        "skipped":     len(skipped),
        "total_chars": sum(r.char_count for r in ok),
        "sources":     [asdict(r) for r in results],
        "data_audit": {
            "found": [
                {"source": r.source_id, "dimension": r.dimension,
                 "chars": r.char_count, "url": r.url}
                for r in ok
            ],
            "missing": [
                {"source": r.source_id, "dimension": r.dimension,
                 "url": r.url, "reason": r.notes}
                for r in failed
            ],
            "gap_impact": {
                "cost_economics":    "Incomplete if CERC tariff orders or SECI auction data missing — CAPEX estimates become indicative only. Missing KERC orders limits Karnataka cost comparison.",
                "grid_access":       "Missing Open Access Registry data means queue wait times are estimated from regulations, not actuals. BESCOM data gap limits Karnataka grid congestion analysis.",
                "subsidies_policy":  "Missing MNRE scheme PDFs limits subsidy amount precision — ballpark from NITI Aayog still possible. Missing KREDL data limits Karnataka scheme detail.",
                "utility_standards": "Missing state RPO orders (RERC/GERC/KERC) means only national targets available — state compliance gap unknown for all three states.",
                "approval_signals":  "EIA transcripts require manual download — NLP over this dimension will be thinner than others. Pavagada land-lease model is unique unknown unknown for Karnataka.",
                "unknown_unknowns":  "Prayas + JMK data provides reasonable national coverage. Karnataka curtailment data and Pavagada farmer-lease model are key discovery-layer findings.",
            },
        },
    }

    out = META_DIR / "metadata.json"
    out.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    log.info(f"Metadata written → {out}")
    return meta


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PowerTrust India — Layer 1 Ingestion")
    parser.add_argument("--dim",   help="Only ingest one dimension (e.g. costs, grid, subsidies)")
    parser.add_argument("--state", help="Only ingest one state (rajasthan, gujarat, karnataka, national)")
    parser.add_argument("--test",  action="store_true", help="Dry run — print URLs, no downloads")
    args = parser.parse_args()

    dim_map = {
        "costs":     "cost_economics",
        "grid":      "grid_access",
        "subsidies": "subsidies_policy",
        "utility":   "utility_standards",
        "approvals": "approval_signals",
        "unknown":   "unknown_unknowns",
    }

    sources = SOURCES
    if args.dim:
        key = dim_map.get(args.dim, args.dim)
        sources = [s for s in sources if s.dimension == key]
        if not sources:
            log.error(f"No sources found for dimension '{args.dim}'. "
                      f"Valid keys: {list(dim_map.keys())}")
            return
    if args.state:
        st = args.state.lower()
        sources = [s for s in sources
                   if s.state_scope == st or s.state_scope in ("national", "all_three")]
        if not sources:
            log.error(f"No sources found for state '{args.state}'. "
                      f"Valid: rajasthan, gujarat, karnataka, national")
            return

    log.info(f"Starting ingestion — {len(sources)} sources "
             f"({'dry run' if args.test else 'live'})")

    results = []
    for i, rec in enumerate(sources, 1):
        log.info(f"[{i}/{len(sources)}] {rec.dimension} / {rec.source_id}")
        rec = ingest_source(rec, dry_run=args.test)
        results.append(rec)
        if not args.test and rec.status == "ok":
            time.sleep(DELAY)   # rate-limit to avoid 429s

    # Summary
    ok_count      = sum(1 for r in results if r.status == "ok")
    failed_count  = sum(1 for r in results if r.status == "failed")
    total_chars   = sum(r.char_count for r in results)

    log.info("=" * 55)
    log.info(f"Done.  OK: {ok_count}  |  Failed: {failed_count}  |  "
             f"Total chars: {total_chars:,}")

    if not args.test:
        meta = write_metadata(results)
        # Print quick audit summary
        if meta["data_audit"]["missing"]:
            log.warning("Missing sources (check manually):")
            for m in meta["data_audit"]["missing"]:
                log.warning(f"  - {m['source']}  ({m['reason']})")

    return results


if __name__ == "__main__":
    main()
