import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
import os
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
from groq import Groq
import chromadb
from insights_dashboard import render_insights_tab, load_structured_data, _best_value

# ── Config ────────────────────────────────────────────────────────────────────
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data", "raw")

# Read API key from Streamlit secrets (cloud) or environment variable (local)
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

USEFUL_FILES = [
    "rerc_tariff_rajasthan_full.txt",
    "kerc_tariff_karnataka_full.txt",
    "cea_installed_capacity_full.txt",
    "gerc_tariff_gujarat.txt",
    "cea_national_electricity_plan.txt",
    "cerc_rpo_regulations.txt",
    "mnre_pm_surya_ghar.txt",
    "jmk_solar_report_2025.txt",
    "renewable_power_generation_full.txt",
]

# ── RAG Pipeline ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_rag():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    chroma_client = chromadb.Client()
    try:
        chroma_client.delete_collection("solar_india")
    except:
        pass
    collection = chroma_client.get_or_create_collection(
        name="solar_india",
        metadata={"hnsw:space": "cosine"}
    )

    def clean_text(text):
        text = text.replace("\t", " ")
        text = re.sub(r" +", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    def paragraph_chunk(filepath, filename, max_words=300):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        text = clean_text(text)
        if "gujarat" in filename:
            state = "Gujarat"
        elif "rajasthan" in filename:
            state = "Rajasthan"
        elif "karnataka" in filename:
            state = "Karnataka"
        elif "installed_capacity" in filename:
            state = "Rajasthan Gujarat Karnataka National India"
        else:
            state = "National India"

        paragraphs = text.split("\n\n")
        chunks, current_chunk, current_words = [], [], 0
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            para_words = len(para.split())
            if current_words + para_words > max_words:
                if current_chunk:
                    chunks.append(f"[State: {state}] " + " ".join(current_chunk))
                current_chunk = para.split()
                current_words = para_words
            else:
                current_chunk.extend(para.split())
                current_words += para_words
        if current_chunk:
            chunks.append(f"[State: {state}] " + " ".join(current_chunk))
        return chunks

    all_chunks, all_metadata = [], []
    for filename in USEFUL_FILES:
        filepath = os.path.join(DATA_FOLDER, filename)
        if os.path.exists(filepath):
            chunks = paragraph_chunk(filepath, filename)
            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadata.append({
                    "source": filename,
                    "state": "Gujarat" if "gujarat" in filename else
                             "Rajasthan" if "rajasthan" in filename else
                             "Karnataka" if "karnataka" in filename else
                             "Rajasthan Gujarat Karnataka National" if "installed_capacity" in filename else
                             "National"
                })

    batch_size = 50
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        meta  = all_metadata[i:i + batch_size]
        ids   = [f"chunk_{j}" for j in range(i, i + len(batch))]
        embeddings = embedder.encode(batch).tolist()
        collection.add(documents=batch, embeddings=embeddings,
                       metadatas=meta, ids=ids)

    groq_client = Groq(api_key=GROQ_API_KEY)
    return embedder, collection, groq_client


def ask_question(question, embedder, collection, groq_client,
                 chat_history=None, n_results=10):
    query_embedding = embedder.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=n_results)

    context = ""
    sources = []
    for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
        context += f"\n---\nSource: {metadata['source']} | State: {metadata['state']}\n{doc}\n"
        sources.append(metadata["source"])

    messages = [
        {"role": "system", "content":
            "You are a solar energy expert for India. Only answer from provided context. "
            "Never hallucinate. Format in clean markdown."}
    ]
    if chat_history:
        for turn in chat_history[-6:]:
            messages.append({"role": turn["role"], "content": turn["content"]})

    prompt = f"""You are an expert solar energy analyst specializing in India's renewable energy sector.

You have access to documents from:
- Gujarat (GERC Tariff Order 2025)
- Rajasthan (RERC Tariff Order 2025)
- Karnataka (KERC Tariff Order 2025)
- National Grid (CEA National Electricity Plan 2022-32)
- National Capacity (CEA Installed Capacity Report 2025)
- National RPO (CERC RPO Regulations 2022)
- National Subsidies (MNRE PM Surya Ghar Guidelines)
- Global Market Data (IRENA Renewable Power Generation Costs 2023)
- India Market Data (JMK Solar Report 2025)

STRICT RULES:
1. Answer ONLY using context provided below
2. NEVER use your own knowledge or make up numbers
3. Always cite exact source document
4. If you see RPO % numbers list ALL of them by year
5. If you see a table reproduce it clearly
6. If question asks about specific state focus on that state
7. If data not found say exactly: "⚠️ Not available in current dataset"
8. Be specific - include actual numbers, percentages, costs
9. For comparison questions cover ALL 3 states

FORMATTING RULES:
- Use ## for main heading
- Use ### for sub headings
- Use **bold** for important numbers
- Use bullet points for lists
- Use markdown tables for comparisons
- Cite the source document name inline within the answer where relevant
- Do NOT add a separate source block or citation at the end of the response

CONTEXT FROM DOCUMENTS:
{context}

QUESTION: {question}

THINK ABOUT WHAT TYPE OF QUESTION THIS IS AND RESPOND ACCORDINGLY:
- Cost/CAPEX → show Rs. crore/MW or USD/kW with years
- Tariff/Rates → show Rs./kWh rates
- RPO/Targets → show yearly targets as markdown table
- Capacity/MW → show state wise comparison table
- Subsidies → list all amounts and conditions
- Rules/Regulations → bullet points with charges
- Comparison → markdown table with all 3 states
- General → structured bullet points

Provide detailed accurate answer in markdown format:"""

    messages.append({"role": "user", "content": prompt})
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.0
    )
    return response.choices[0].message.content, list(set(sources))


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PowerTrust Solar Intelligence",
    page_icon="🌞",
    layout="wide"
)

with st.spinner("Loading RAG pipeline..."):
    embedder, collection, groq_client = load_rag()

st.sidebar.title("🌞 PowerTrust")
st.sidebar.markdown("**India Solar Intelligence**")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "📊 Visual Report",
    "💬 Chat",
    "📈 Insights & Scoring",
    "🌍 Scalability",
    "📋 Data Audit"
])

# ── Chart data ────────────────────────────────────────────────────────────────
@st.cache_data
def get_visual_data():
    from insights_dashboard import load_state_csv
    csv_data = load_state_csv()
    raw = load_structured_data()

    def csv_val(state, col, fallback):
        try:
            return float(csv_data.get(state, {}).get(col, fallback) or fallback)
        except (ValueError, TypeError):
            return fallback

    return {
        "re_capacity": {
            "Rajasthan": csv_val("Rajasthan", "total_re_mw",  39995.99),
            "Gujarat":   csv_val("Gujarat",   "total_re_mw",  38029.77),
            "Karnataka": csv_val("Karnataka", "total_re_mw",  21810.51),
        },
        "tariffs": {
            "Rajasthan (existing)": 3.34,
            "Rajasthan (new)":      csv_val("Rajasthan", "avg_tariff_rs_kwh", 3.21),
            "Gujarat (rooftop)":    2.32,
            "Gujarat (utility)":    csv_val("Gujarat",   "avg_tariff_rs_kwh", 3.45),
        },
        "capex_cr_per_mw": round(csv_val("Rajasthan", "capex_lakh_per_mw", 380) / 100, 1),
        "gujarat_rpo": {
            "Year":          ["2025-26", "2026-27", "2027-28", "2028-29", "2029-30"],
            "Total RPO (%)": [33.01, 35.95, 38.81, 41.37, 43.33],
        },
    }

# ============================================
# PAGE 1: VISUAL REPORT
# ============================================
if page == "📊 Visual Report":
    vd = get_visual_data()
    st.title("🌞 India Solar Intelligence Report")
    st.markdown("**Rajasthan • Gujarat • Karnataka**")
    st.caption("Data sourced from CERC, RERC, GERC, KERC, MNRE, CEA — ingested and extracted automatically.")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Solar CAPEX", f"₹{vd['capex_cr_per_mw']} Cr/MW", "CERC benchmark 2024")
    with col2:
        st.metric("Global LCOE", "USD 0.044/kWh", "-90% since 2010")
    with col3:
        st.metric("India RE Capacity", "197,201 MW", "CEA Sep 2025")
    with col4:
        st.metric("Solar Tariff Rajasthan", "₹2.61/kWh", "2024 auction")
    st.markdown("---")

    st.subheader("📊 Total Renewable Energy Capacity by State")
    st.caption("Source: CEA Installed Capacity Report Sep 2025")
    cap_df = pd.DataFrame({"State": list(vd["re_capacity"].keys()),
                            "RE Capacity (MW)": list(vd["re_capacity"].values())})
    fig1 = px.bar(cap_df, x="State", y="RE Capacity (MW)", color="State",
                  title="Installed Renewable Energy Capacity (As on 30.09.2025)",
                  color_discrete_sequence=["#FF6B35", "#004E89", "#1A936F"],
                  text="RE Capacity (MW)")
    fig1.update_traces(texttemplate="%{text:,.0f} MW", textposition="outside")
    fig1.update_layout(showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("---")

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("📈 Gujarat RPO Targets (2025–2030)")
        st.caption("Source: GERC MYT Tariff Order 2025")
        fig2 = px.line(pd.DataFrame(vd["gujarat_rpo"]), x="Year", y="Total RPO (%)",
                       title="Gujarat RPO trajectory — 33% → 43% by 2030",
                       markers=True, color_discrete_sequence=["#004E89"])
        fig2.add_hline(y=33, line_dash="dash", line_color="gray",
                       annotation_text="2025-26 baseline", annotation_position="right")
        st.plotly_chart(fig2, use_container_width=True)
    with col_r:
        st.subheader("💰 Solar Tariff Comparison")
        st.caption("Source: RERC, GERC tariff orders")
        fig3 = px.bar(pd.DataFrame({"Category": list(vd["tariffs"].keys()),
                                    "Tariff (Rs./kWh)": list(vd["tariffs"].values())}),
                      x="Category", y="Tariff (Rs./kWh)",
                      title="Solar tariff rates by state and segment",
                      color="Category",
                      color_discrete_sequence=["#FF6B35","#FF9F1C","#004E89","#2EC4B6"],
                      text="Tariff (Rs./kWh)")
        fig3.update_traces(texttemplate="₹%{text}", textposition="outside")
        fig3.update_layout(showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
    st.markdown("---")

    st.subheader("📉 CAPEX trend — India solar")
    st.caption("Source: IRENA, MNRE benchmark documents")
    fig4 = px.line(pd.DataFrame({"Year": [2010,2013,2016,2019,2021,2023,2024],
                                  "CAPEX (Cr/MW)": [18.0,12.0,7.5,5.5,4.5,4.2,4.0]}),
                   x="Year", y="CAPEX (Cr/MW)",
                   title="India utility-scale solar CAPEX decline (₹ Cr/MW)",
                   markers=True, color_discrete_sequence=["#E07B39"])
    fig4.add_scatter(x=[2024], y=[4.0], mode="markers+text",
                     text=["₹4.0 Cr/MW (2024)"], textposition="top right",
                     marker=dict(size=12, color="#E07B39"), showlegend=False)
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown("---")

    st.subheader("🌍 Global solar LCOE trend (2010–2023)")
    st.caption("Source: IRENA Renewable Power Generation Costs 2023")
    fig5 = px.line(pd.DataFrame({"Year": [2010,2012,2014,2016,2018,2020,2021,2022,2023],
                                  "LCOE (USD/kWh)": [0.460,0.320,0.220,0.100,0.079,0.057,0.048,0.049,0.044]}),
                   x="Year", y="LCOE (USD/kWh)",
                   title="Solar PV global weighted average LCOE — 90% drop in 13 years",
                   markers=True, color_discrete_sequence=["#1A936F"])
    fig5.add_hline(y=0.100, line_dash="dash", line_color="gray",
                   annotation_text="Fossil fuel cost range", annotation_position="right")
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown("---")

    col6, col7 = st.columns(2)
    with col6:
        st.subheader("🔵 Rajasthan RPO breakdown FY2025-26")
        st.caption("Source: RERC Discoms Tariff Order FY2025-26")
        fig6 = px.pie(pd.DataFrame({"Category": ["Solar (Other RPO)","Wind RPO","HPO"],
                                    "Percentage": [28.17,3.36,1.48]}),
                      values="Percentage", names="Category",
                      title="Rajasthan RPO composition — Total 33.01%",
                      color_discrete_sequence=["#FF6B35","#004E89","#1A936F"])
        fig6.update_traces(textinfo="label+percent")
        st.plotly_chart(fig6, use_container_width=True)
    with col7:
        st.subheader("💰 PM Surya Ghar subsidy structure")
        st.caption("Source: MNRE PM Surya Ghar Guidelines 2024")
        fig7 = px.bar(pd.DataFrame({"Capacity": ["First 2 kWp","Additional kWp (2-3 kW)","Above 3 kW"],
                                    "Subsidy (Rs/kWp)": [30000,18000,0]}),
                      x="Capacity", y="Subsidy (Rs/kWp)",
                      title="Central Financial Assistance (CFA) per kWp",
                      color="Capacity",
                      color_discrete_sequence=["#FF6B35","#FF9F1C","#CCCCCC"],
                      text="Subsidy (Rs/kWp)")
        fig7.update_traces(texttemplate="₹%{text:,}", textposition="outside")
        fig7.update_layout(showlegend=False, yaxis=dict(range=[0,35000]))
        st.plotly_chart(fig7, use_container_width=True)
    st.markdown("---")

    st.subheader("📋 State comparison — key metrics")
    st.caption("Source: RERC, GERC, KERC tariff orders; CEA installed capacity report")
    st.dataframe(pd.DataFrame({
        "Metric":    ["Installed Solar (GW)","Total RE (MW)","Avg Tariff (Rs/kWh)",
                      "CAPEX (Lakh/MW)","Total RPO Target (%)","Approval Time (months)"],
        "Rajasthan": ["28.5","39,996","3.21","380","33.01%","14"],
        "Gujarat":   ["20.1","38,030","3.45","375","33.01%","10"],
        "Karnataka": ["9.9","21,811","5.10","390","33.10%","12"],
    }), use_container_width=True, hide_index=True)
    st.markdown("---")

    col9, col10 = st.columns(2)
    with col9:
        st.subheader("⚡ National RE mix — Solar vs Wind")
        st.caption("Source: CEA All India Installed Capacity Report Sep 2025")
        fig9 = px.pie(pd.DataFrame({"Source": ["Solar PV","Wind","Small Hydro","Biomass/Others"],
                                    "Capacity (MW)": [127332,53123,5133,11613]}),
                      values="Capacity (MW)", names="Source",
                      title="India RE installed capacity mix (Sep 2025)",
                      color_discrete_sequence=["#FF6B35","#004E89","#1A936F","#AAAAAA"])
        fig9.update_traces(textinfo="label+percent")
        st.plotly_chart(fig9, use_container_width=True)
    with col10:
        st.subheader("🌐 India vs Global solar CAPEX")
        st.caption("Source: IRENA 2023, MNRE benchmark, CERC SM-2024")
        fig10 = px.bar(pd.DataFrame({
            "Region": ["India CAPEX\n(₹ Cr/MW)","Global CAPEX\n(USD/kW)","India LCOE\n(₹/kWh)","Global LCOE\n(USD cents/kWh)"],
            "Value":  [4.0, 7.58, 2.61, 4.4],
            "Type":   ["India","Global","India","Global"],
        }), x="Region", y="Value", color="Type",
            title="India vs Global cost benchmarks",
            color_discrete_sequence=["#FF6B35","#004E89"], text="Value")
        fig10.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        st.plotly_chart(fig10, use_container_width=True)

# ============================================
# PAGE 2: CHAT
# ============================================
elif page == "💬 Chat":
    st.title("💬 Ask About India Solar")
    st.markdown("Ask any question about building solar farms in India")
    st.caption("Answers grounded in ingested regulatory documents — RERC, GERC, KERC, CEA, MNRE, IRENA")
    st.markdown("---")

    with st.expander("💡 Sample questions to try"):
        st.markdown("""
**Dimension queries**
- What is the solar RPO target for Gujarat in 2025-26?
- What subsidies are available under PM Surya Ghar scheme?
- What is the global benchmark cost for utility-scale solar PV?

**Comparison queries**
- How do the RPO compliance levels compare between Gujarat and Karnataka?
- What are the risks of building solar in Karnataka vs Rajasthan?

**Risk & gap queries**
- What data is missing that would affect our CAPEX estimate?
- What are the risks of building solar in Karnataka?

**Try a follow-up after any answer above to test multi-turn context**
        """)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about solar energy in India..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                answer, sources = ask_question(
                    prompt, embedder, collection, groq_client,
                    chat_history=st.session_state.messages[:-1]
                )
                st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

    if st.session_state.messages:
        if st.button("🗑️ Clear chat"):
            st.session_state.messages = []
            st.rerun()

# ============================================
# PAGE 3: INSIGHTS & SCORING
# ============================================
elif page == "📈 Insights & Scoring":
    render_insights_tab()

# ============================================
# PAGE 4: SCALABILITY
# ============================================
elif page == "🌍 Scalability":
    st.title("🌍 Adding New Countries")
    st.markdown("---")
    st.subheader("How PowerTrust Scales")
    st.markdown("""
    Adding a new country requires **no code changes** — just data:
    
    ### Steps to Add a New Country:
    1. **Collect documents** — regulatory orders, tariff documents, grid data
    2. **Run ingest pipeline** — `python layer1_ingest.py` chunks and embeds automatically
    3. **Re-embed** — `python layer2_embed.py` rebuilds the vector index
    4. **Select country** — system immediately answers questions for the new country
    """)
    st.table(pd.DataFrame({
        "Country": ["🇮🇳 India"],
        "States/Regions": ["Rajasthan, Gujarat, Karnataka"],
        "Documents": [23],
        "Status": ["✅ Active"]
    }))
    st.markdown("### Coming Soon:")
    st.table(pd.DataFrame({
        "Country": ["🇲🇾 Malaysia", "🇲🇽 Mexico", "🇧🇷 Brazil"],
        "Status":  ["📋 Planned",   "📋 Planned", "📋 Planned"]
    }))

# ============================================
# PAGE 5: DATA AUDIT
# ============================================
elif page == "📋 Data Audit":
    st.title("📋 Data Availability Audit")
    st.markdown("---")
    st.subheader("✅ Data Found and Used")
    st.table(pd.DataFrame({
        "Document": [
            "RERC Tariff Order 2025","KERC Tariff Order 2025","GERC Tariff Order 2025",
            "CEA National Electricity Plan","CEA Installed Capacity Report",
            "CERC RPO Regulations","MNRE PM Surya Ghar Guidelines",
            "IRENA Renewable Power Costs 2023","JMK Solar Report 2025"
        ],
        "State/Source": [
            "Rajasthan","Karnataka","Gujarat","National","National",
            "National","National","Global","India Market"
        ],
        "Key Data": [
            "Solar tariff Rs.2.61/kWh, RPO 33%",
            "RPO compliance 33.10%, wheeling charges",
            "RPO targets 33%-43.33%, solar rates",
            "Solar CAPEX Rs.4.5 Cr/MW, grid plans",
            "State capacity: RJ=39995, GJ=38029, KA=21810 MW",
            "National RPO regulations and trajectory",
            "Subsidy Rs.30,000/kWp for first 2 kWp",
            "LCOE USD 0.044/kWh, CAPEX USD 758/kW",
            "23.8 GW solar installed FY2025"
        ]
    }))
    st.markdown("---")
    st.subheader("❌ Data Missing or Inaccessible")
    st.table(pd.DataFrame({
        "Data": [
            "Karnataka solar tariff (Rs./kWh)","SECI auction results 2024",
            "Grid interconnection wait times","Open Access Registry data",
            "EIA public hearing transcripts"
        ],
        "Reason": [
            "Not extracted from KERC tables","URL returned 404",
            "POSOCO site blocked from Canada","DNS resolution failure",
            "MoEFCC page returned 404"
        ],
        "Impact on Output": [
            "Karnataka cost comparison uses KERC order estimate",
            "No discovered auction price benchmarks",
            "Queue wait times are indicative only",
            "Open access approval data unavailable",
            "Approval signals dimension has thin coverage"
        ]
    }))
    st.markdown("---")
    st.subheader("⚙️ Data pipeline summary")
    st.markdown("""
- **Layer 1** — 41 sources attempted, 23 successfully ingested (6.7M chars)
- **Layer 2** — 3,252 chunks embedded, 4.8MB FAISS index
- **Sources** — RERC, GERC, KERC, CEA, CERC, MNRE, IRENA, JMK Research
- **Gaps** — Karnataka government sites geoblocked from Canada; documented above
    """)


# ── Load real data once at startup ────────────────────────────────────────────
@st.cache_data
def get_visual_data():
    """Load chart data from CSV + JSON. CSV for table data, JSON for RPO."""
    from insights_dashboard import load_state_csv, load_structured_data, _best_value
    csv_data = load_state_csv()
    raw      = load_structured_data()
    r        = raw.get("rpo_targets", [])

    def csv_val(state, col, fallback):
        try:
            return float(csv_data.get(state, {}).get(col, fallback) or fallback)
        except (ValueError, TypeError):
            return fallback

    return {
        "re_capacity": {
            "Rajasthan": csv_val("Rajasthan", "total_re_mw",  39995.99),
            "Gujarat":   csv_val("Gujarat",   "total_re_mw",  38029.77),
            "Karnataka": csv_val("Karnataka", "total_re_mw",  21810.51),
        },
        "tariffs": {
            "Rajasthan (existing)": 3.34,
            "Rajasthan (new)":      csv_val("Rajasthan", "avg_tariff_rs_kwh", 3.21),
            "Gujarat (rooftop)":    2.32,
            "Gujarat (utility)":    csv_val("Gujarat",   "avg_tariff_rs_kwh", 3.45),
        },
        "capex_cr_per_mw": round(csv_val("Rajasthan", "capex_lakh_per_mw", 380) / 100, 1),
        "india_re_mw":     197201,
        # RPO trajectory confirmed from GERC MYT Tariff Order 2025
        "gujarat_rpo": {
            "Year":          ["2025-26", "2026-27", "2027-28", "2028-29", "2029-30"],
            "Total RPO (%)": [33.01, 35.95, 38.81, 41.37, 43.33],
        },
    }
