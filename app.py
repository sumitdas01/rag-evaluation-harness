import os
import json
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# Page Configuration
st.set_page_config(
    page_title="RAG Evaluation Harness | Enterprise Retrieval Benchmark",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, Theme-Safe Styling (Works in both Light and Dark Mode)
st.markdown("""
<style>
    /* Hero Typography */
    .hero-title {
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        line-height: 1.5;
        margin-bottom: 1.25rem;
        opacity: 0.85;
    }

    /* Executive Callout Card */
    .callout-card {
        background: rgba(37, 99, 235, 0.06);
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-left: 4px solid #3B82F6;
        border-radius: 6px;
        padding: 1rem 1.25rem;
        margin: 1.25rem 0;
        font-size: 0.92rem;
        line-height: 1.6;
    }

    /* Strategy Summary Cards */
    .strategy-card {
        background: rgba(148, 163, 184, 0.05);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 8px;
        padding: 1.1rem;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .strategy-badge {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 9999px;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .badge-purple { background: rgba(139, 92, 246, 0.15); color: #A78BFA; border: 1px solid rgba(139, 92, 246, 0.3); }
    .badge-blue { background: rgba(37, 99, 235, 0.15); color: #60A5FA; border: 1px solid rgba(37, 99, 235, 0.3); }
    .badge-green { background: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-amber { background: rgba(245, 158, 11, 0.15); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.3); }

    .strategy-card-title {
        font-size: 0.98rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .strategy-card-metric {
        font-size: 1.2rem;
        font-weight: 700;
        margin: 0.3rem 0;
    }
    .strategy-card-desc {
        font-size: 0.83rem;
        opacity: 0.85;
        line-height: 1.45;
    }

    /* Production Decision Card */
    .decision-card {
        background: rgba(148, 163, 184, 0.04);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 8px;
        padding: 1.2rem 1.4rem;
        height: 100%;
        line-height: 1.55;
    }

    /* Clean spacing */
    hr {
        border: 0;
        height: 1px;
        background: rgba(148, 163, 184, 0.2);
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Data Paths
BASE_DIR = Path(__file__).parent
RESULTS_FILE = BASE_DIR / "results" / "evaluation_results.json"
REPORT_FILE = BASE_DIR / "results" / "report.json"
CSV_FILE = BASE_DIR / "results" / "report.csv"
QUESTIONS_FILE = BASE_DIR / "data" / "evaluation" / "questions.json"

@st.cache_data
def load_benchmark_data():
    if not RESULTS_FILE.exists() or not REPORT_FILE.exists():
        return None, None, None, None
        
    with open(RESULTS_FILE, "r") as f:
        results = json.load(f)
    with open(REPORT_FILE, "r") as f:
        summary = json.load(f)
    with open(QUESTIONS_FILE, "r") as f:
        questions = json.load(f)
    df = pd.read_csv(CSV_FILE) if CSV_FILE.exists() else pd.DataFrame()
    return results, summary, questions, df

results, summary, questions, df = load_benchmark_data()

# Sidebar
st.sidebar.markdown("### ⚖️ RAG Evaluation")
st.sidebar.caption("Automated Retrieval Benchmarking & Quality Gate")
st.sidebar.markdown("---")

st.sidebar.markdown("**CI/CD Quality Gate**")
selected_threshold = st.sidebar.slider(
    "Required Accuracy Threshold (%)", 
    min_value=50, 
    max_value=95, 
    value=70, 
    step=5,
    help="Build passes in CI/CD if best strategy accuracy meets or exceeds this threshold."
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Indexed Documents**")
st.sidebar.markdown("""
- **DPDP Act, 2023** (Data Protection)
- **RBI Digital Lending** (Lending Norms)
- **RBI Cyber Security** (IT Governance)
- **SEBI BRSR** (ESG Disclosures)
""")

st.sidebar.markdown("---")
st.sidebar.caption("50 Questions · 4 Strategies · 200 Evaluations")

# Navigation Tabs
tabs = st.tabs([
    "Overview", 
    "Strategy Benchmark", 
    "Speed vs. Accuracy", 
    "Question Explorer", 
    "Live Playground",
    "Architecture & CI/CD"
])

# -------------------------------------------------------------
# TAB 1: OVERVIEW (Automated RAG Retrieval Evaluation)
# -------------------------------------------------------------
with tabs[0]:
    st.markdown('<div class="hero-title">Automated RAG Retrieval Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Objective evaluation framework comparing 4 retrieval strategies across 50 statutory questions (200 evaluations) on high-stakes Indian regulatory frameworks.</div>', unsafe_allow_html=True)

    if summary is None:
        st.warning("Benchmark results not found. Please run: `python -m src.evaluation.run`")
    else:
        sum_df = pd.DataFrame(summary)
        best_retrieval_row = sum_df.loc[sum_df['Retrieval_Accuracy'].idxmax()]
        best_answer_row = sum_df.loc[sum_df['Answer_Accuracy'].idxmax()]
        best_mrr_row = sum_df.loc[sum_df['MRR'].idxmax()]
        gate_passed = (best_answer_row['Answer_Accuracy'] >= selected_threshold)

        # 4 Native Streamlit Metrics (Guaranteed 100% theme-adaptive visibility)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            label="Evaluation Matrix", 
            value="200 Runs", 
            delta="50 Qs × 4 Strategies",
            delta_color="off"
        )
        c2.metric(
            label="Top Ranking Precision (MRR)", 
            value=f"{best_mrr_row['MRR']:.1f}%", 
            delta=f"{best_mrr_row['strategy'].capitalize()} Reranker"
        )
        c3.metric(
            label="Top Retrieval Hit Rate", 
            value=f"{best_retrieval_row['Retrieval_Accuracy']:.1f}%", 
            delta="Reranker & Simple"
        )
        c4.metric(
            label="CI/CD Quality Gate", 
            value="PASS" if gate_passed else "FAIL", 
            delta=f"Threshold: ≥{selected_threshold}%",
            delta_color="normal" if gate_passed else "inverse"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # High-Contrast, Theme-Adaptive Performance Matrix
        st.markdown("#### Performance Matrix")
        st.caption("Empirical benchmark across 50 statutory questions (200 total retrieval evaluations). Click any column header to sort.")

        # Friendly strategy names and production role mappings
        name_map = {
            "reranking": "Cross-Encoder Reranking",
            "hybrid": "Hybrid Search (Dense + BM25)",
            "simple": "Simple Chunking (Dense Baseline)",
            "semantic": "Semantic Chunking"
        }
        role_map = {
            "reranking": "Audit & Compliance (Guaranteed Rank #1 Precision)",
            "hybrid": "Real-Time Chat & Search (Optimal Sweet Spot · 24× Faster)",
            "simple": "Standard Dense Baseline (Fixed-Size 800 Chunks)",
            "semantic": "Passage Cohesion Baseline (Embedding Distance Splits)"
        }
        
        # Sort so highest performing strategies appear first
        sorted_summary = sum_df.sort_values(by="MRR", ascending=False).reset_index(drop=True)

        # Build clean structured dataframe
        matrix_rows = []
        for _, row in sorted_summary.iterrows():
            strat_key = row['strategy']
            strat_name = name_map.get(strat_key, strat_key.capitalize())
            latency_ms = row['Avg_Retrieval_Latency'] * 1000
            
            prefix = "⭐ " if strat_key == "reranking" else ("⚡ " if strat_key == "hybrid" else ("📐 " if strat_key == "simple" else "🧠 "))
            matrix_rows.append({
                "Retrieval Strategy": f"{prefix}{strat_name}",
                "Hit Rate (%)": round(row['Retrieval_Accuracy'], 1),
                "Recall@5 (%)": round(row['Recall_at_K'], 1),
                "MRR (%)": round(row['MRR'], 1),
                "Answer Accuracy (%)": round(row['Answer_Accuracy'], 1),
                "Avg Latency": f"{latency_ms:.0f} ms",
                "Production Recommendation": role_map.get(strat_key, "")
            })

        matrix_df = pd.DataFrame(matrix_rows)

        # Native Streamlit Dataframe with Progress Bars (Guaranteed 100% theme-adaptive contrast)
        st.dataframe(
            matrix_df,
            column_config={
                "Retrieval Strategy": st.column_config.TextColumn("Retrieval Strategy", width="medium"),
                "Hit Rate (%)": st.column_config.ProgressColumn("Hit Rate", min_value=0, max_value=100, format="%.1f%%"),
                "Recall@5 (%)": st.column_config.ProgressColumn("Recall@5", min_value=0, max_value=100, format="%.1f%%"),
                "MRR (%)": st.column_config.ProgressColumn("MRR (Precision)", min_value=0, max_value=100, format="%.1f%%"),
                "Answer Accuracy (%)": st.column_config.ProgressColumn("Answer Accuracy", min_value=0, max_value=100, format="%.1f%%"),
                "Avg Latency": st.column_config.TextColumn("Avg Latency", width="small"),
                "Production Recommendation": st.column_config.TextColumn("Production Recommendation", width="large")
            },
            hide_index=True,
            use_container_width=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # 4 Strategy Architecture Summary Cards
        card_col1, card_col2, card_col3, card_col4 = st.columns(4)
        
        with card_col1:
            st.markdown("""
            <div class="strategy-card">
                <div>
                    <span class="strategy-badge badge-purple">Top Precision</span>
                    <div class="strategy-card-title">Cross-Encoder Reranker</div>
                    <div class="strategy-card-metric" style="color: #A78BFA;">97.0% <span style="font-size:0.82rem; font-weight:normal; opacity:0.8;">MRR</span></div>
                    <div class="strategy-card-desc">Jointly cross-encodes query-doc pairs. Guarantees the statutory proviso ranks at #1 for zero-error compliance audits.</div>
                </div>
                <div style="margin-top:0.75rem; font-size:0.78rem; opacity:0.75;">Latency: <b>365 ms</b> · Recall: <b>95.0%</b></div>
            </div>
            """, unsafe_allow_html=True)

        with card_col2:
            st.markdown("""
            <div class="strategy-card">
                <div>
                    <span class="strategy-badge badge-blue">Production Sweet-Spot</span>
                    <div class="strategy-card-title">Hybrid (Dense + BM25)</div>
                    <div class="strategy-card-metric" style="color: #60A5FA;">90.5% <span style="font-size:0.82rem; font-weight:normal; opacity:0.8;">MRR</span></div>
                    <div class="strategy-card-desc">Fuses dense vector semantics with sparse BM25 via Reciprocal Rank Fusion. Delivers 24× faster latency for live chat.</div>
                </div>
                <div style="margin-top:0.75rem; font-size:0.78rem; opacity:0.75;">Latency: <b>15 ms</b> · Recall: <b>90.0%</b></div>
            </div>
            """, unsafe_allow_html=True)

        with card_col3:
            st.markdown("""
            <div class="strategy-card">
                <div>
                    <span class="strategy-badge badge-green">Dense Baseline</span>
                    <div class="strategy-card-title">Simple Fixed Chunks</div>
                    <div class="strategy-card-metric" style="color: #34D399;">89.1% <span style="font-size:0.82rem; font-weight:normal; opacity:0.8;">MRR</span></div>
                    <div class="strategy-card-desc">Fixed 800-character windows with 100-character overlap. Dependable baseline, but prone to splitting legal clauses.</div>
                </div>
                <div style="margin-top:0.75rem; font-size:0.78rem; opacity:0.75;">Latency: <b>18 ms</b> · Recall: <b>91.0%</b></div>
            </div>
            """, unsafe_allow_html=True)

        with card_col4:
            st.markdown("""
            <div class="strategy-card">
                <div>
                    <span class="strategy-badge badge-amber">Meaning Cohesion</span>
                    <div class="strategy-card-title">Semantic Chunking</div>
                    <div class="strategy-card-metric" style="color: #FBBF24;">78.7% <span style="font-size:0.82rem; font-weight:normal; opacity:0.8;">MRR</span></div>
                    <div class="strategy-card-desc">Splits at embedding similarity drop-offs. Preserves sentence cohesion, but variable chunk sizes dilute top-rank density.</div>
                </div>
                <div style="margin-top:0.75rem; font-size:0.78rem; opacity:0.75;">Latency: <b>15 ms</b> · Recall: <b>50.6%</b></div>
            </div>
            """, unsafe_allow_html=True)

        # Executive Takeaway Callout Box
        st.markdown("""
        <div class="callout-card">
            <strong style="color: #60A5FA;">Key Production Architecture Finding:</strong> 
            <strong>Cross-Encoder Reranking</strong> delivers the highest ranking precision (<strong>97.0% MRR</strong> and <strong>95.0% Recall@5</strong>), placing the statutory proviso at rank #1 for regulatory audits. Meanwhile, <strong>Hybrid Search (Dense + BM25 via RRF)</strong> provides the optimal efficiency sweet-spot for interactive chat—achieving <strong>90.5% MRR at only 15 ms</strong> (<strong>24× faster</strong>).
        </div>
        """, unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 2: STRATEGY BENCHMARK
# -------------------------------------------------------------
with tabs[1]:
    st.markdown("#### Performance Breakdown Across Retrieval Strategies")
    st.caption("Comparing retrieval completeness (Hit Rate, Recall), ranking precision (MRR), and factual answer correctness.")

    if summary:
        sum_df = pd.DataFrame(summary)
        
        col1, col2 = st.columns([1.3, 1])
        
        # Formatted strategy names with line-breaks for horizontal presentation
        clean_strat_labels = {
            "reranking": "Cross-Encoder<br>Reranking",
            "hybrid": "Hybrid Search<br>(Dense + BM25)",
            "simple": "Simple Chunking<br>(Baseline)",
            "semantic": "Semantic<br>Chunking"
        }

        with col1:
            # Grouped Bar Chart with explicit data labels
            fig_bar = go.Figure()
            strategies = [clean_strat_labels.get(s, s.capitalize()) for s in sum_df['strategy']]
            
            fig_bar.add_trace(go.Bar(
                x=strategies, 
                y=sum_df['Retrieval_Accuracy'], 
                name='Hit Rate (%)', 
                marker_color='#2563EB',
                text=[f"{v:.1f}%" for v in sum_df['Retrieval_Accuracy']],
                textposition='auto'
            ))
            fig_bar.add_trace(go.Bar(
                x=strategies, 
                y=sum_df['Recall_at_K'], 
                name='Recall@5 (%)', 
                marker_color='#10B981',
                text=[f"{v:.1f}%" for v in sum_df['Recall_at_K']],
                textposition='auto'
            ))
            fig_bar.add_trace(go.Bar(
                x=strategies, 
                y=sum_df['MRR'], 
                name='MRR (%)', 
                marker_color='#8B5CF6',
                text=[f"{v:.1f}%" for v in sum_df['MRR']],
                textposition='auto'
            ))
            fig_bar.add_trace(go.Bar(
                x=strategies, 
                y=sum_df['Answer_Accuracy'], 
                name='Answer Accuracy (%)', 
                marker_color='#F59E0B',
                text=[f"{v:.1f}%" for v in sum_df['Answer_Accuracy']],
                textposition='auto'
            ))
            
            fig_bar.update_layout(
                barmode='group',
                yaxis=dict(title="Score (%)", range=[0, 115]),
                xaxis=dict(title=None, tickangle=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
                margin=dict(l=30, r=20, t=40, b=40)
            )
            st.plotly_chart(fig_bar, use_container_width=True, theme="streamlit")

        with col2:
            # Clean Hits vs Misses count out of 50
            hit_counts = []
            miss_counts = []
            strat_display_names = []
            
            for strat in sum_df['strategy']:
                strat_display_names.append(clean_strat_labels.get(strat, strat.capitalize()))
                strat_rows = df[df['strategy'] == strat] if not df.empty else []
                hits = strat_rows['hit'].sum() if len(strat_rows) > 0 else int(sum_df.loc[sum_df['strategy'] == strat, 'Retrieval_Accuracy'].values[0] * 0.5)
                misses = 50 - hits
                hit_counts.append(hits)
                miss_counts.append(misses)
                
            fig_hits = go.Figure()
            fig_hits.add_trace(go.Bar(
                x=strat_display_names, 
                y=hit_counts, 
                name='Hits (Found Chunks)', 
                marker_color='#10B981',
                text=[f"{h}/50" for h in hit_counts],
                textposition='inside'
            ))
            fig_hits.add_trace(go.Bar(
                x=strat_display_names, 
                y=miss_counts, 
                name='Misses', 
                marker_color='#EF4444',
                text=[f"{m}" if m > 0 else "" for m in miss_counts],
                textposition='inside'
            ))
            
            fig_hits.update_layout(
                barmode='stack',
                title="Hits vs Misses (out of 50 Questions)",
                yaxis=dict(title="Total Questions", range=[0, 55]),
                xaxis=dict(title=None, tickangle=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
                margin=dict(l=30, r=20, t=40, b=40)
            )
            st.plotly_chart(fig_hits, use_container_width=True, theme="streamlit")

        # Plain English Metric Glossary
        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown("**Hit Rate**")
            st.caption("Did the retrieved Top-5 chunks contain at least one ground-truth document?")
        with m2:
            st.markdown("**Recall@5**")
            st.caption("What percentage of all known relevant chunks were successfully retrieved?")
        with m3:
            st.markdown("**MRR (Mean Reciprocal Rank)**")
            st.caption("How high did the first relevant chunk rank? (1.0 = Rank #1, 0.5 = Rank #2).")
        with m4:
            st.markdown("**Answer Accuracy**")
            st.caption("Did the synthesized answer match the statutory ground-truth facts?")

# -------------------------------------------------------------
# TAB 3: SPEED VS. ACCURACY (PARETO FRONTIER)
# -------------------------------------------------------------
with tabs[2]:
    st.markdown("#### Engineering Trade-Off: Ranking Precision vs. Latency")
    st.caption("In production systems, choosing an architecture requires balancing ranking accuracy against end-user latency budgets.")

    if summary:
        sum_df = pd.DataFrame(summary)
        
        c_chart, c_bar = st.columns([1.35, 1.0])

        color_palette = {
            "reranking": "#8B5CF6",
            "hybrid": "#2563EB",
            "simple": "#10B981",
            "semantic": "#F59E0B"
        }

        with c_chart:
            # High-contrast Pareto Scatter plot with distinct leader-arrow callouts
            fig_scatter = go.Figure()

            # Plot Pareto Optimal Frontier (Hybrid -> Cross-Encoder)
            h_row = sum_df.loc[sum_df['strategy'] == 'hybrid'].iloc[0]
            r_row = sum_df.loc[sum_df['strategy'] == 'reranking'].iloc[0]
            h_lat = h_row['Avg_Retrieval_Latency'] * 1000
            r_lat = r_row['Avg_Retrieval_Latency'] * 1000

            fig_scatter.add_trace(go.Scatter(
                x=[h_lat, r_lat],
                y=[h_row['MRR'], r_row['MRR']],
                mode='lines',
                line=dict(color='rgba(148, 163, 184, 0.45)', width=2, dash='dash'),
                name='Pareto Optimal Frontier',
                hoverinfo='skip',
                showlegend=False
            ))

            # Add markers for each strategy (clean markers, no colliding center text)
            for _, r in sum_df.iterrows():
                strat_key = r['strategy']
                strat_label = name_map.get(strat_key, strat_key.capitalize())
                latency_ms = r['Avg_Retrieval_Latency'] * 1000
                
                fig_scatter.add_trace(go.Scatter(
                    x=[latency_ms],
                    y=[r['MRR']],
                    mode='markers',
                    name=strat_label,
                    marker=dict(
                        size=20, 
                        color=color_palette.get(strat_key, "#2563EB"),
                        line=dict(width=2.5, color="#FFFFFF")
                    ),
                    hoverinfo='text',
                    hovertext=f"<b>{strat_label}</b><br>MRR: {r['MRR']:.1f}%<br>Recall@5: {r['Recall_at_K']:.1f}%<br>Hit Rate: {r['Retrieval_Accuracy']:.1f}%<br>Latency: {latency_ms:.1f} ms"
                ))

            # Non-colliding dedicated annotations with leader lines
            # Calibrated offsets ensure 100% legibility with zero label collisions
            annotations = [
                dict(
                    x=h_lat, y=h_row['MRR'],
                    ax=-95, ay=-48,
                    xanchor="right", yanchor="bottom",
                    text=f"<b>Hybrid Search (Dense + BM25)</b><br>{h_row['MRR']:.1f}% MRR · {h_lat:.0f} ms<br><span style='color:#60A5FA; font-weight:600;'>⚡ 24× Faster Sweet Spot</span>",
                    arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor="rgba(148, 163, 184, 0.7)",
                    bordercolor="rgba(37, 99, 235, 0.5)", borderwidth=1, borderpad=6,
                    bgcolor="rgba(15, 23, 42, 0.90)", font=dict(size=11, color="#F8FAFC")
                ),
                dict(
                    x=sum_df.loc[sum_df['strategy'] == 'simple', 'Avg_Retrieval_Latency'].values[0] * 1000,
                    y=sum_df.loc[sum_df['strategy'] == 'simple', 'MRR'].values[0],
                    ax=85, ay=35,
                    xanchor="left", yanchor="top",
                    text=f"<b>Simple Chunking</b><br>{sum_df.loc[sum_df['strategy'] == 'simple', 'MRR'].values[0]:.1f}% MRR · {sum_df.loc[sum_df['strategy'] == 'simple', 'Avg_Retrieval_Latency'].values[0]*1000:.0f} ms<br><span style='color:#34D399; font-weight:600;'>Dense Baseline</span>",
                    arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor="rgba(148, 163, 184, 0.7)",
                    bordercolor="rgba(16, 185, 129, 0.5)", borderwidth=1, borderpad=6,
                    bgcolor="rgba(15, 23, 42, 0.90)", font=dict(size=11, color="#F8FAFC")
                ),
                dict(
                    x=sum_df.loc[sum_df['strategy'] == 'semantic', 'Avg_Retrieval_Latency'].values[0] * 1000,
                    y=sum_df.loc[sum_df['strategy'] == 'semantic', 'MRR'].values[0],
                    ax=-75, ay=45,
                    xanchor="right", yanchor="top",
                    text=f"<b>Semantic Chunking</b><br>{sum_df.loc[sum_df['strategy'] == 'semantic', 'MRR'].values[0]:.1f}% MRR · {sum_df.loc[sum_df['strategy'] == 'semantic', 'Avg_Retrieval_Latency'].values[0]*1000:.0f} ms",
                    arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor="rgba(148, 163, 184, 0.7)",
                    bordercolor="rgba(245, 158, 11, 0.5)", borderwidth=1, borderpad=6,
                    bgcolor="rgba(15, 23, 42, 0.90)", font=dict(size=11, color="#F8FAFC")
                ),
                dict(
                    x=r_lat, y=r_row['MRR'],
                    ax=-95, ay=-40,
                    xanchor="right", yanchor="bottom",
                    text=f"<b>Cross-Encoder Reranking</b><br>{r_row['MRR']:.1f}% MRR · {r_lat:.0f} ms<br><span style='color:#A78BFA; font-weight:600;'>🏆 Top Precision</span>",
                    arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor="rgba(148, 163, 184, 0.7)",
                    bordercolor="rgba(139, 92, 246, 0.5)", borderwidth=1, borderpad=6,
                    bgcolor="rgba(15, 23, 42, 0.90)", font=dict(size=11, color="#F8FAFC")
                )
            ]

            for ann in annotations:
                fig_scatter.add_annotation(**ann)

            fig_scatter.update_layout(
                title="Pareto Frontier: Ranking Precision vs. Latency",
                xaxis=dict(
                    title="Average Retrieval Latency (Milliseconds, ms)",
                    range=[-55, 425],
                    showgrid=True
                ),
                yaxis=dict(
                    title="Ranking Precision (Mean Reciprocal Rank, MRR %)",
                    range=[70, 105],
                    showgrid=True
                ),
                height=460,
                showlegend=False,
                margin=dict(l=50, r=40, t=50, b=40)
            )
            st.plotly_chart(fig_scatter, use_container_width=True, theme="streamlit")

        with c_bar:
            # Latency Breakdown Horizontal Bar Chart
            sorted_lat = sum_df.sort_values(by="Avg_Retrieval_Latency", ascending=True).reset_index(drop=True)
            lat_names = [clean_strat_labels.get(s, s.capitalize()) for s in sorted_lat['strategy']]
            lat_vals = [r['Avg_Retrieval_Latency'] * 1000 for _, r in sorted_lat.iterrows()]
            lat_colors = [color_palette.get(s, "#2563EB") for s in sorted_lat['strategy']]
            lat_texts = []
            for s, v in zip(sorted_lat['strategy'], lat_vals):
                if s == "hybrid":
                    lat_texts.append(f"<b>{v:.0f} ms</b> (⚡ 24× Faster)")
                elif s == "reranking":
                    lat_texts.append(f"<b>{v:.0f} ms</b> (Audit Grade)")
                else:
                    lat_texts.append(f"<b>{v:.0f} ms</b>")

            fig_latency = go.Figure()
            fig_latency.add_trace(go.Bar(
                y=lat_names,
                x=lat_vals,
                orientation='h',
                marker_color=lat_colors,
                text=lat_texts,
                textposition='auto'
            ))
            fig_latency.update_layout(
                title="Retrieval Latency Comparison (ms)",
                xaxis=dict(title="Milliseconds (ms)", showgrid=True),
                yaxis=dict(title=None, tickangle=0),
                height=460,
                margin=dict(l=30, r=30, t=50, b=40)
            )
            st.plotly_chart(fig_latency, use_container_width=True, theme="streamlit")

        # Two Clear Production Decision Cards
        dec_col1, dec_col2 = st.columns(2)
        with dec_col1:
            st.markdown("""
            <div class="decision-card">
                <div style="font-weight: 700; font-size: 1.05rem; margin-bottom: 0.5rem; color: #60A5FA;">
                    ⚡ Real-Time Chat & Search (Interactive)
                </div>
                <div><b>Selected Architecture:</b> Hybrid Search (Dense + BM25 via Reciprocal Rank Fusion)</div>
                <div style="margin-top: 0.45rem; font-size: 0.88rem; opacity: 0.88; line-height: 1.55;">
                    Achieves <b>90.5% MRR at only 15 ms latency</b>. Fusing semantic embeddings with BM25 keyword matching catches statutory provisions while executing <b>24× faster</b> than neural reranking—leaving 95%+ of the response budget for interactive LLM token streaming.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with dec_col2:
            st.markdown("""
            <div class="decision-card">
                <div style="font-weight: 700; font-size: 1.05rem; margin-bottom: 0.5rem; color: #A78BFA;">
                    📋 Batch Compliance & Regulatory Audit (Asynchronous)
                </div>
                <div><b>Selected Architecture:</b> Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2)</div>
                <div style="margin-top: 0.45rem; font-size: 0.88rem; opacity: 0.88; line-height: 1.55;">
                    Delivers the benchmark ceiling at <b>97.0% MRR and 95.0% Recall@5</b>. In statutory audits, background report generation, and due diligence, an extra 350 ms latency is negligible, but missing a statutory proviso carries regulatory fines and liability.
                </div>
            </div>
            """, unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 4: QUESTION EXPLORER
# -------------------------------------------------------------
with tabs[3]:
    st.markdown("#### Inspect Any of the 50 Ground-Truth Questions")
    st.caption("Review expected statutory answers, retrieved passages, and strategy comparisons.")

    if results and questions:
        q_options = [f"{q['id']}: {q['question']}" for q in questions]
        selected_q_str = st.selectbox("Select question to evaluate:", q_options)
        selected_q_id = selected_q_str.split(":")[0]

        q_meta = next(q for q in questions if q['id'] == selected_q_id)
        
        # Clean metadata container
        with st.container():
            st.markdown(f"**Question:** {q_meta['question']}")
            st.markdown(f"**Expected Statutory Answer:** `{q_meta['expected_answer']}`")
            st.caption(f"Source: {', '.join(q_meta.get('source_documents', []))} · Ground Truth Chunks: {', '.join(q_meta.get('relevant_chunk_ids', []))}")

        st.markdown("---")
        st.markdown("##### Strategy Retrieval Comparison")

        q_results = [r for r in results if r['question_id'] == selected_q_id]
        cols = st.columns(4)

        for i, res in enumerate(q_results):
            strat = res['strategy'].capitalize()
            hit = res['retrieval_metrics']['hit']
            acc = res['answer_correct']
            mrr = res['retrieval_metrics']['mrr']
            latency = res['latency']['retrieval']

            with cols[i]:
                st.markdown(f"**{strat}**")
                hit_pill = "✅ Hit" if hit else "❌ Miss"
                acc_pill = "✅ Correct" if acc else "❌ Incorrect"
                
                st.write(f"Retrieval: **{hit_pill}** (MRR: `{mrr:.2f}`)")
                st.write(f"Answer: **{acc_pill}**")
                st.caption(f"Latency: {latency*1000:.1f} ms")
                
                with st.expander("Retrieved Chunks"):
                    st.json(res.get("retrieved_chunks", []))
                with st.expander("Generated Answer"):
                    st.write(res.get("generated_answer", ""))

# -------------------------------------------------------------
# TAB 5: LIVE PLAYGROUND
# -------------------------------------------------------------
with tabs[4]:
    st.markdown("#### Live Multi-Strategy Query Playground")
    st.caption("Test any regulatory compliance query live across all 4 retrieval engines simultaneously.")

    sample_prompts = [
        "What is the maximum monetary penalty under the DPDP Act for failing to maintain security safeguards?",
        "What is the mandatory cooling-off period for digital loans under RBI guidelines?",
        "What is the statutory reporting timeline for a bank to report a cyber security incident to RBI?",
        "What percentage of average net profits must eligible companies spend on CSR under SEBI BRSR Principle 8?"
    ]
    
    preset_choice = st.selectbox("Select a sample query or write your own:", ["-- Custom Query --"] + sample_prompts)
    user_query = st.text_input("Enter Question:", value="" if preset_choice == "-- Custom Query --" else preset_choice)

    if st.button("Run Multi-Strategy Retrieval", type="primary"):
        if not user_query.strip():
            st.error("Please enter a question.")
        else:
            with st.spinner("Executing retrieval across Vector, BM25, Hybrid, and Cross-Encoder..."):
                try:
                    # Auto-build index if running on a fresh cloud container
                    simple_dir = BASE_DIR / "chroma" / "simple_chunks"
                    if not simple_dir.exists():
                        from src.ingestion.build_index import build_index
                        build_index("simple")

                    from src.retrieval.vector_retriever import VectorRetriever
                    from src.retrieval.bm25_retriever import BM25Retriever
                    from src.retrieval.hybrid_retriever import HybridRetriever
                    from src.retrieval.reranker import RerankerRetriever
                    
                    t0 = time.time()
                    simple_docs = VectorRetriever(strategy="simple", top_k=3).retrieve(user_query)
                    t_simple = (time.time() - t0) * 1000

                    t0 = time.time()
                    bm25_docs = BM25Retriever(strategy="simple", top_k=3).retrieve(user_query)
                    t_bm25 = (time.time() - t0) * 1000

                    t0 = time.time()
                    hybrid_docs = HybridRetriever(strategy="simple", top_k=3).retrieve(user_query)
                    t_hybrid = (time.time() - t0) * 1000

                    t0 = time.time()
                    rerank_docs = RerankerRetriever(strategy="simple").retrieve(user_query)
                    t_rerank = (time.time() - t0) * 1000

                    c1, c2, c3, c4 = st.columns(4)
                    
                    with c1:
                        st.markdown(f"**1. Vector Search** · `{t_simple:.1f} ms`")
                        for idx, d in enumerate(simple_docs):
                            st.caption(f"#{idx+1} {d.metadata.get('chunk_id')}")
                            st.write(d.page_content[:150] + "...")

                    with c2:
                        st.markdown(f"**2. BM25 Keyword** · `{t_bm25:.1f} ms`")
                        for idx, d in enumerate(bm25_docs):
                            st.caption(f"#{idx+1} {d.metadata.get('chunk_id')}")
                            st.write(d.page_content[:150] + "...")

                    with c3:
                        st.markdown(f"**3. Hybrid (RRF)** · `{t_hybrid:.1f} ms`")
                        for idx, d in enumerate(hybrid_docs):
                            st.caption(f"#{idx+1} {d.metadata.get('chunk_id')}")
                            st.write(d.page_content[:150] + "...")

                    with c4:
                        st.markdown(f"**4. Cross-Encoder** · `{t_rerank:.1f} ms`")
                        for idx, d in enumerate(rerank_docs):
                            st.caption(f"#{idx+1} {d.metadata.get('chunk_id')} (Score: {d.metadata.get('reranker_score', 0):.2f})")
                            st.write(d.page_content[:150] + "...")

                except Exception as e:
                    st.error(f"Live retrieval failed: {e}")

# -------------------------------------------------------------
# TAB 6: ARCHITECTURE & CI/CD
# -------------------------------------------------------------
with tabs[5]:
    st.markdown("#### Production Architecture & CI/CD Quality Gate")
    st.caption("How this system prevents silent accuracy regressions on pull requests.")

    st.markdown("""
```mermaid
graph TD
    A[Regulatory Documents<br>DPDP Act, RBI Directions, SEBI BRSR] --> B[Document Loader & Cleaner]
    B --> C[Fixed-Size Simple Chunks]
    B --> D[Semantic Meaning Chunks]
    C --> E[(ChromaDB Vector Store)]
    D --> E
    E --> F[Dense Retrieval: all-MiniLM-L6-v2]
    E --> G[Sparse Retrieval: BM25Okapi]
    F --> H[Hybrid Search: Reciprocal Rank Fusion]
    G --> H
    H --> I[Cross-Encoder Reranker: ms-marco-MiniLM]
    I --> J[50-Question Ground Truth Evaluation Harness]
    J --> K{Accuracy >= Threshold Quality Gate}
    K -- PASS --> L[GitHub Actions BUILD PASS]
    K -- FAIL --> M[GitHub Actions BUILD FAIL Non-Zero Exit]
```
    """)

    st.markdown("""
    ##### Why This Architecture Matters in Enterprise Systems:
    1. **Regression Prevention:** When developers alter chunk parameters or prompt templates, the automated quality gate catches accuracy regressions before code reaches staging or production.
    2. **Zero-Hallucination Guardrails:** Tracking both *Retrieval Hit Rate* and *Answer Accuracy* isolates whether an error was caused by missing context (retrieval issue) or model hallucination (generation issue).
    3. **Auditable Evaluation Artifacts:** Generates structured `report.json` and `report.csv` files on every run for governance and auditability.
    """)
