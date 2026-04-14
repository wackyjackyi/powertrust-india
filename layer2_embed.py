"""
PowerTrust India — Layer 2: Chunking + Embedding
=================================================
Reads raw text files from data/raw/, chunks them, embeds with
sentence-transformers, and builds a FAISS index.

Usage:
    python layer2_embed.py            # full run
    python layer2_embed.py --test     # embed first 3 sources only

Output:
    data/solar_india.index    FAISS vector index
    data/chunks_meta.pkl      chunk text + metadata list
    data/structured_data.json key numbers extracted for dashboard
"""

import os
import re
import json
import pickle
import argparse
import logging
from pathlib import Path
from datetime import date

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False
    print("[WARN] sentence-transformers not installed. Run: pip install sentence-transformers")

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("[WARN] faiss-cpu not installed. Run: pip install faiss-cpu")

# ── Config ────────────────────────────────────────────────────────────────────
RAW_DIR       = Path("data/raw")
META_FILE     = Path("data/metadata.json")
INDEX_FILE    = Path("data/solar_india.index")
CHUNKS_FILE   = Path("data/chunks_meta.pkl")
STRUCTURED_FILE = Path("data/structured_data.json")

CHUNK_SIZE    = 400    # words per chunk
OVERLAP       = 80     # word overlap between chunks
MODEL_NAME    = "all-MiniLM-L6-v2"   # fast, good quality, 384-dim

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, source_meta: dict,
               chunk_size: int = CHUNK_SIZE,
               overlap: int = OVERLAP) -> list[dict]:
    """
    Split text into overlapping word-windows.
    Each chunk carries full source metadata for attribution.
    """
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()

    if len(words) < 30:
        return []   # too short to be useful

    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        window = words[i: i + chunk_size]
        if len(window) < 30:
            break
        chunk_text_str = " ".join(window)
        chunks.append({
            "text":         chunk_text_str,
            "source_id":    source_meta.get("source_id", "unknown"),
            "source_name":  source_meta.get("source_name", ""),
            "url":          source_meta.get("url", ""),
            "dimension":    source_meta.get("dimension", ""),
            "state_scope":  source_meta.get("state_scope", "national"),
            "chunk_index":  len(chunks),
            "word_start":   i,
            "date_accessed": source_meta.get("date_accessed", str(date.today())),
        })
    return chunks


# ── Structured data extractor ─────────────────────────────────────────────────

# Keywords that must appear near a number for it to count as solar-specific.
# This prevents biomass/biogas/MSW values from being captured.
SOLAR_KEYWORDS = [
    "solar", "pv", "photovoltaic", "rooftop", "rts", "kusum",
    "wind", "renewable", "re ", "seci", "mnre",
]

def _is_solar_context(context: str) -> bool:
    """Return True if context is about solar/RE, not biomass/biogas/MSW/coal."""
    ctx = context.lower()
    # Reject if clearly about wrong technology
    reject = ["biomass", "biogas", "msw", "municipal solid waste",
              "coal", "gas", "thermal", "nuclear", "hydro",
              "refuse derived", "bagasse", "co-generation", "cogen"]
    if any(r in ctx for r in reject):
        return False
    # Accept if solar/RE keyword present
    return any(k in ctx for k in SOLAR_KEYWORDS)


def extract_structured_data(chunks: list[dict]) -> dict:
    """
    Pull solar-specific numbers out of chunks for the dashboard layer.
    Uses targeted regex with solar keyword filtering.
    """
    structured = {
        "capex_benchmarks":  [],
        "tariff_rates":      [],
        "rpo_targets":       [],
        "capacity_stats":    [],
        "subsidy_amounts":   [],
        "_generated":        str(date.today()),
        "_source_count":     len(set(c["source_id"] for c in chunks)),
        "_chunk_count":      len(chunks),
    }

    # Solar-specific patterns only
    patterns = {
        "capex_benchmarks": [
            # e.g. "capital cost of solar PV ... Rs. 375 lakh/MW"
            r"(?:solar|pv|photovoltaic)[^.]{0,120}?(?:capital cost|capex)[^\d]{0,30}?([\d,]+(?:\.\d+)?)\s*(?:lakh|crore)[^\n]{0,30}?(?:MW|kW)",
            r"(?:capital cost|capex)[^.]{0,60}?(?:solar|pv)[^\d]{0,30}?([\d,]+(?:\.\d+)?)\s*(?:lakh|crore)[^\n]{0,30}?(?:MW|kW)",
            # benchmark cost Rs/Wp e.g. "benchmark cost ... 54 Rs/Wp"
            r"benchmark\s+cost[^\d]{0,40}?([\d,]+(?:\.\d+)?)\s*(?:Rs\.?|INR)[^\n]{0,20}?(?:Wp|kWp|kW)",
        ],
        "tariff_rates": [
            # Discovered solar tariff e.g. "₹2.61/kWh" or "tariff of Rs. 2.61 per kWh"
            r"(?:solar|pv|rooftop|wind)[^.]{0,80}?(?:tariff|rate|discovered)[^\d]{0,20}?(?:Rs\.?|₹)\s*([\d.]+)\s*(?:per\s+)?(?:kWh|unit)",
            r"(?:tariff|rate)[^\d]{0,20}?(?:Rs\.?|₹)\s*([\d.]+)\s*(?:per\s+)?(?:kWh|unit)[^.]{0,60}?(?:solar|pv|wind|renewable)",
            # levelised tariff for solar specifically
            r"levelis[e]?d\s+tariff[^.]{0,60}?(?:solar|wind|pv)[^\d]{0,30}?(?:Rs\.?|₹)\s*([\d.]+)",
            r"(?:solar|wind|pv)[^.]{0,60}?levelis[e]?d\s+tariff[^\d]{0,30}?([\d.]+)\s*(?:Rs\.?|₹|per)",
        ],
        "rpo_targets": [
            # e.g. "solar RPO target of 8%" or "RPO ... 25%"
            r"(?:solar\s+RPO|RPO[^.]{0,40}?solar)[^\d]{0,30}?([\d.]+)\s*%",
            r"(?:RPO|renewable purchase obligation)[^\d]{0,60}?([\d.]+)\s*%",
            r"([\d.]+)\s*%[^\n]{0,60}?(?:RPO|renewable purchase obligation)",
            # e.g. "solar target 25%" from RERC/GERC orders
            r"solar[^\d]{0,40}?target[^\d]{0,20}?([\d.]+)\s*%",
        ],
        "capacity_stats": [
            # State-level solar capacity e.g. "28,500 MW solar" or "28.5 GW"
            r"([\d,]+(?:\.\d+)?)\s*(?:GW|MW)[^\n]{0,60}?(?:solar|pv|photovoltaic)\s*(?:capacity|installed|power)",
            r"(?:solar|pv)\s*(?:capacity|installed)[^\d]{0,30}?([\d,]+(?:\.\d+)?)\s*(?:GW|MW)",
            r"(?:installed\s+capacity)[^\d]{0,30}?([\d,]+(?:\.\d+)?)\s*(?:GW|MW)[^\n]{0,40}?(?:solar|renewable|re)",
        ],
        "subsidy_amounts": [
            # e.g. "CFA of Rs. 30,000 per kW" or "subsidy of Rs 78,000"
            r"(?:CFA|subsidy|central\s+financial\s+assistance)[^\d]{0,30}?(?:Rs\.?|₹)\s*([\d,]+(?:\.\d+)?)\s*(?:per\s+kW|per\s+kWp|lakh)",
            r"(?:Rs\.?|₹)\s*([\d,]+(?:\.\d+)?)\s*(?:per\s+kW|per\s+kWp)[^\n]{0,60}?(?:CFA|subsidy|rooftop|solar)",
        ],
    }

    seen = {k: set() for k in patterns}

    for chunk in chunks:
        text = chunk["text"]
        for field, pats in patterns.items():
            for pat in pats:
                matches = re.findall(pat, text, re.IGNORECASE)
                for m in matches:
                    val = m.replace(",", "")
                    key = f"{val}_{chunk['source_id']}"
                    if key not in seen[field] and len(structured[field]) < 50:
                        # Get surrounding context
                        idx = text.lower().find(m.lower().replace(",",""))
                        context = text[max(0, idx-100):idx+150].strip()
                        # Filter: only keep solar/RE relevant context
                        if not _is_solar_context(context):
                            continue
                        # Filter: sanity-check numeric ranges
                        try:
                            fval = float(val)
                            if field == "capex_benchmarks" and not (50 < fval < 600):
                                continue  # solar CAPEX lakh/MW should be 300-500
                            if field == "tariff_rates" and not (1.0 < fval < 15.0):
                                continue  # Rs/kWh sanity check
                            if field == "rpo_targets" and not (1 < fval < 100):
                                continue
                            if field == "subsidy_amounts" and fval < 100:
                                continue  # too small to be a subsidy amount
                        except ValueError:
                            continue

                        seen[field].add(key)
                        structured[field].append({
                            "value":     val,
                            "context":   context,
                            "source_id": chunk["source_id"],
                            "state":     chunk["state_scope"],
                            "dimension": chunk["dimension"],
                        })

    log.info(f"  Structured data extracted:")
    for k, v in structured.items():
        if isinstance(v, list):
            log.info(f"    {k}: {len(v)} entries")
        if isinstance(v, list):
            log.info(f"    {k}: {len(v)} entries")

    return structured


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PowerTrust India — Layer 2 Embedding")
    parser.add_argument("--test", action="store_true",
                        help="Embed first 3 sources only (quick test)")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--overlap",    type=int, default=OVERLAP)
    args = parser.parse_args()

    if not HAS_ST or not HAS_FAISS:
        log.error("Missing dependencies. Install with: pip install sentence-transformers faiss-cpu")
        return

    # ── Load metadata ──────────────────────────────────────────────────────────
    if not META_FILE.exists():
        log.error(f"metadata.json not found at {META_FILE}. Run layer1_ingest.py first.")
        return

    with open(META_FILE, encoding="utf-8", errors="ignore") as f:
        meta = json.load(f)

    ok_sources = [s for s in meta["sources"] if s["status"] == "ok"]
    log.info(f"Found {len(ok_sources)} successfully ingested sources")

    if args.test:
        ok_sources = ok_sources[:3]
        log.info("Test mode — using first 3 sources only")

    # ── Chunk all sources ──────────────────────────────────────────────────────
    all_chunks = []
    for src in ok_sources:
        txt_path = RAW_DIR / f"{src['source_id']}.txt"
        if not txt_path.exists():
            log.warning(f"  Text file not found: {txt_path}")
            continue

        text = txt_path.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_text(text, src,
                            chunk_size=args.chunk_size,
                            overlap=args.overlap)
        log.info(f"  {src['source_id']}: {len(text):,} chars → {len(chunks)} chunks")
        all_chunks.extend(chunks)

    log.info(f"Total chunks: {len(all_chunks):,}")

    if not all_chunks:
        log.error("No chunks produced. Check data/raw/ has .txt files.")
        return

    # ── Embed ──────────────────────────────────────────────────────────────────
    log.info(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    texts = [c["text"] for c in all_chunks]
    log.info(f"Embedding {len(texts):,} chunks (this takes 1-3 minutes)...")

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    log.info(f"Embeddings shape: {embeddings.shape}")

    # ── Build FAISS index ──────────────────────────────────────────────────────
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))
    log.info(f"FAISS index built: {index.ntotal} vectors, dim={dim}")

    # ── Save outputs ───────────────────────────────────────────────────────────
    Path("data").mkdir(exist_ok=True)

    faiss.write_index(index, str(INDEX_FILE))
    log.info(f"Index saved → {INDEX_FILE} ({INDEX_FILE.stat().st_size / 1024 / 1024:.1f} MB)")

    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(all_chunks, f)
    log.info(f"Chunks saved → {CHUNKS_FILE} ({CHUNKS_FILE.stat().st_size / 1024:.0f} KB)")

    # ── Extract structured data for dashboard ──────────────────────────────────
    log.info("Extracting structured data for dashboard...")
    structured = extract_structured_data(all_chunks)
    with open(STRUCTURED_FILE, "w", encoding="utf-8") as f:
        json.dump(structured, f, indent=2, ensure_ascii=False)
    log.info(f"Structured data saved → {STRUCTURED_FILE}")

    # ── Summary ────────────────────────────────────────────────────────────────
    log.info("=" * 55)
    log.info(f"Layer 2 complete.")
    log.info(f"  Chunks:     {len(all_chunks):,}")
    log.info(f"  Dimensions: {dim}")
    log.info(f"  Index size: {INDEX_FILE.stat().st_size / 1024 / 1024:.1f} MB")
    log.info(f"  Ready for Layer 3 (RAG + Groq)")


if __name__ == "__main__":
    main()
