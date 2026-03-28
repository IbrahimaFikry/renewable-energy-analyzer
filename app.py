"""
app.py
------
Renewable Energy Investment Analyzer — Interactive Dashboard

"""

import re
import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────────
NAVY  = "#151931"
GOLD  = "#D4A96A"
GOLD2 = "#C8914D"

CLUSTER_COLORS = {
    "Ready Markets":      "#1C6B45",
    "Transition Markets": "#C8914D",
    "Watch & Wait":       "#B03A2E",
}
CLUSTER_BG = {
    "Ready Markets":      "rgba(28,107,69,0.07)",
    "Transition Markets": "rgba(200,145,77,0.07)",
    "Watch & Wait":       "rgba(176,58,46,0.07)",
}
REGION_COLORS = {
    "SEA":    "#1A6FA8",
    "Africa": "#8B4513",
    "LatAm":  "#1C6B45",
}
DIM_COLORS = {
    "score_market_opportunity":          "#1A6FA8",
    "score_decarbonization_opportunity": NAVY,
    "score_business_environment":        "#1C6B45",
    "score_energy_security":             "#C8914D",
}
DIM_LABELS = {
    "score_market_opportunity":          "Market Opportunity",
    "score_decarbonization_opportunity": "Decarbonisation",
    "score_business_environment":        "Business Environment",
    "score_energy_security":             "Energy Security",
}
DIM_COLS = list(DIM_LABELS.keys())
DIM_WEIGHTS_DEFAULT = [0.30, 0.45, 0.20, 0.05]

COUNTRY_NAMES = {
    "VNM": "Vietnam",      "IDN": "Indonesia",    "PHL": "Philippines",
    "THA": "Thailand",     "MYS": "Malaysia",     "KHM": "Cambodia",
    "BGD": "Bangladesh",   "KEN": "Kenya",        "NGA": "Nigeria",
    "GHA": "Ghana",        "ETH": "Ethiopia",     "TZA": "Tanzania",
    "SEN": "Senegal",      "CIV": "Côte d'Ivoire","MAR": "Morocco",
    "BRA": "Brazil",       "MEX": "Mexico",       "COL": "Colombia",
    "PER": "Peru",         "CHL": "Chile",
}

CARD_STYLE = {
    "background": "white",
    "borderRadius": "4px",
    "padding": "28px",
    "boxShadow": "0 1px 4px rgba(0,0,0,0.08)",
    "border": "1px solid #EBEBEB",
}

LAYOUT_BASE = dict(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(family="Arial, sans-serif", color="#3D3D3D", size=12),
    hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial, sans-serif"),
)

# ── Helpers ────────────────────────────────────────────────────────────────

def strip_emoji(text: str) -> str:
    """Remove emoji characters from a string."""
    return re.sub(r"[^\w\s&'./,\-]", "", text).strip()

def annotate(fig, title, subtitle=""):
    """Add a bold title + grey subtitle annotation above the chart."""
    anns = list(fig.layout.annotations or [])
    anns.append(dict(
        text=f"<b>{title}</b>",
        xref="paper", yref="paper",
        x=0, y=1.10, xanchor="left", yanchor="bottom",
        font=dict(size=13, color=NAVY, family="Arial, sans-serif"),
        showarrow=False,
    ))
    if subtitle:
        anns.append(dict(
            text=subtitle,
            xref="paper", yref="paper",
            x=0, y=1.04, xanchor="left", yanchor="bottom",
            font=dict(size=10.5, color="#888888", family="Arial, sans-serif"),
            showarrow=False,
        ))
    fig.update_layout(annotations=anns)
    return fig

def card(children, extra_style=None):
    style = {**CARD_STYLE, **(extra_style or {})}
    return html.Div(children, style=style)

def section_label(text):
    return html.Div(text, style={
        "fontSize": "10px", "fontWeight": "700", "letterSpacing": "2px",
        "textTransform": "uppercase", "color": GOLD2, "marginBottom": "6px",
    })

def section_title(text):
    return html.H2(text, style={
        "fontSize": "18px", "fontWeight": "700", "color": NAVY,
        "margin": "0 0 4px 0", "letterSpacing": "-0.2px",
    })

def section_subtitle(text):
    return html.P(text, style={
        "fontSize": "13px", "color": "#888888", "margin": "0 0 20px 0",
    })

def header_block(label, title, subtitle=None):
    return html.Div([
        section_label(label),
        section_title(title),
        section_subtitle(subtitle) if subtitle else None,
    ])

# ── Data Loading ───────────────────────────────────────────────────────────

def load_data():
    try:
        clusters = pd.read_csv("outputs/market_clusters.csv", index_col="country_code")
        sensitivity = pd.read_csv("outputs/sensitivity_analysis.csv", index_col="country_code")

        clusters["country_name"] = clusters.index.map(COUNTRY_NAMES)
        sensitivity["country_name"] = sensitivity.index.map(COUNTRY_NAMES)

        # Normalise cluster labels
        clusters["cluster_label"] = clusters["cluster_label"].apply(strip_emoji)
        if "stability" in sensitivity.columns:
            sensitivity = sensitivity.rename(columns={"stability": "rank_stability"})

        # Ensure numeric types
        for col in clusters.columns:
            if col not in ["country_name", "region", "cluster_label"]:
                clusters[col] = pd.to_numeric(clusters[col], errors="coerce")
        for col in sensitivity.columns:
            if col not in ["country_name", "region", "rank_stability"]:
                sensitivity[col] = pd.to_numeric(sensitivity[col], errors="coerce")

        return clusters, sensitivity
    except FileNotFoundError:
        return pd.DataFrame(), pd.DataFrame()

clusters_df, sensitivity_df = load_data()

# ──  Fallback static data  ──────────────────────────────────

FALLBACK = pd.DataFrame({
    "country_code": ["MYS","CHL","THA","VNM","MEX","IDN","GHA","SEN","CIV","BGD",
                     "PER","MAR","COL","PHL","BRA","KHM","TZA","NGA","ETH","KEN"],
    "country_name": ["Malaysia","Chile","Thailand","Vietnam","Mexico","Indonesia",
                     "Ghana","Senegal","Côte d'Ivoire","Bangladesh","Peru","Morocco",
                     "Colombia","Philippines","Brazil","Cambodia","Tanzania","Nigeria",
                     "Ethiopia","Kenya"],
    "region":       ["SEA","LatAm","SEA","SEA","LatAm","SEA","Africa","Africa",
                     "Africa","SEA","LatAm","Africa","LatAm","SEA","LatAm","SEA",
                     "Africa","Africa","Africa","Africa"],
    "total_score":  [86.56,70.28,60.59,60.12,53.67,53.19,51.06,50.53,49.16,48.14,
                     47.73,46.89,46.23,46.16,43.28,42.30,41.20,30.70,30.60,23.90],
    "rank":         list(range(1, 21)),
    "cluster_label": ["Ready Markets"]*2 + ["Transition Markets"]*15 + ["Watch & Wait"]*3,
    "score_market_opportunity":
        [66.11,60.24,34.65,53.38,35.59,36.68,36.77,37.33,42.81,37.58,
         35.69,25.81,41.07,32.99,44.29,33.56,29.50,23.19,42.77,31.00],
    "score_decarbonization_opportunity":
        [97.30,60.61,73.51,62.48,71.07,65.92,53.65,49.04,54.99,64.07,
         51.81,55.77,53.53,47.92,58.74,35.89,17.59,17.24,16.82,28.65],
    "score_business_environment":
        [98.97,99.99,62.69,59.36,36.98,58.92,67.98,61.47,40.83,19.80,
         57.75,57.10,53.22,45.84,52.08,32.64,42.50,0.14,13.40,33.52],
    "score_energy_security":
        [62.87,98.76,91.47,82.33,72.31,14.84,45.90,99.33,68.20,81.56,
         52.42,92.07,12.58,54.27,37.77,77.24,81.84,40.53,76.01,45.03],
}).set_index("country_code")

#  sensitivity fallback data
FALLBACK_SENSITIVITY = pd.DataFrame({
    "country_name":        ["Malaysia","Chile","Thailand","Vietnam","Mexico","Indonesia",
                            "Ghana","Senegal","Côte d'Ivoire","Bangladesh","Peru","Morocco",
                            "Colombia","Philippines","Brazil","Cambodia","Tanzania","Nigeria",
                            "Ethiopia","Kenya"],
    "region":              ["SEA","LatAm","SEA","SEA","LatAm","SEA","Africa","Africa",
                            "Africa","SEA","LatAm","Africa","LatAm","SEA","LatAm","SEA",
                            "Africa","Africa","Africa","Africa"],
    "avg_rank":            [1.0,2.0,3.2,3.8,6.8,6.8,7.0,7.8,9.2,9.8,10.2,10.8,11.2,12.5,13.2,13.8,14.8,18.5,18.8,19.0],
    "rank_std":            [0.0,0.0,0.5,0.8,1.5,0.8,0.8,0.8,1.3,1.3,0.0,1.3,1.5,1.3,3.1,0.8,0.8,0.0,1.3,1.3],
    "rank_min":            [1,2,3,4,5,6,6,7,8,9,10,9,10,11,11,13,14,18,17,17],
    "rank_max":            [1,2,4,5,8,7,8,9,11,11,10,12,13,14,16,14,15,18,20,20],
    "rank_balanced":       [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
    "rank_growth_focused": [1,2,4,3,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,5],
    "rank_impact_first":   [1,2,3,5,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
    "rank_risk_averse":    [1,2,3,4,11,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20],
    "rank_stability":      ["Very High","Very High","Very High","Very High","High","Very High",
                            "Very High","Very High","Very High","Very High","Very High","Very High",
                            "High","High","Medium","Very High","Very High","Very High","Very High","Very High"],
})

if clusters_df.empty:
    clusters_df = FALLBACK.copy()
if sensitivity_df.empty:
    sensitivity_df = FALLBACK_SENSITIVITY.copy()

plot_df = clusters_df.reset_index()
src_df  = clusters_df

# ── Chart functions ────────────────────────────────────────────────────────

def chart_rankings(df, custom_weights=None):
    """Horizontal bar chart — all 20 markets, coloured by archetype."""
    if custom_weights:
        df = df.copy()
        df["total_score"] = sum(
            df[col] * w for col, w in zip(DIM_COLS, custom_weights)
        )
        df["rank"] = df["total_score"].rank(ascending=False, method="min").astype(int)

    # Sort ascending (lowest score first)
    sorted_df = df.reset_index().sort_values("total_score", ascending=True).reset_index(drop=True)

    fig = go.Figure()

    # Cluster background bands
    band_colors = {
        'Ready Markets':      'rgba(28,107,69,0.06)',
        'Transition Markets': 'rgba(200,145,77,0.06)',
        'Watch & Wait':       'rgba(176,58,46,0.06)',
    }
    for cluster_lbl, band_color in band_colors.items():
        rows = sorted_df[sorted_df['cluster_label'] == cluster_lbl].index
        if len(rows) == 0:
            continue
        fig.add_shape(
            type='rect', xref='paper', yref='y',
            x0=0, x1=1, y0=rows.min() - 0.5, y1=rows.max() + 0.5,
            fillcolor=band_color, line=dict(width=0), layer='below',
        )

    # Bars
    for cluster_lbl, color in CLUSTER_COLORS.items():
        mask = sorted_df['cluster_label'] == cluster_lbl
        sub = sorted_df[mask]
        if sub.empty:
            continue
        fig.add_trace(go.Bar(
            x=sub['total_score'],
            y=sub['country_name'],
            orientation='h',
            name=cluster_lbl,
            marker=dict(color=color, opacity=0.88),
            text=[f"{v:.1f}" for v in sub['total_score']],
            textposition='outside',
            textfont=dict(size=10.5, color='#444444'),
            hovertemplate=(
                '<b>%{y}</b><br>'
                'Score: %{x:.1f}<br>'
                'Rank: #%{customdata[0]}<br>'
                'Archetype: %{customdata[1]}'
                '<extra></extra>'
            ),
            customdata=list(zip(sub['rank'], sub['cluster_label'])),
        ))

    # Add score = 50 reference line
    fig.add_shape(type="line", x0=50, x1=50, y0=-0.5, y1=19.5,
                  line=dict(color=GOLD2, width=1.5, dash="dot"))
    fig.add_annotation(x=51, y=19.5, text="Score = 50", showarrow=False,
                       xanchor="left", font=dict(size=10, color=GOLD2))

    # Gap bracket between Chile and Thailand
    chile_score = sorted_df.loc[sorted_df['country_name'] == 'Chile', 'total_score'].values[0]
    thailand_score = sorted_df.loc[sorted_df['country_name'] == 'Thailand', 'total_score'].values[0]
    chile_y = sorted_df[sorted_df['country_name'] == 'Chile'].index[0]
    thailand_y = sorted_df[sorted_df['country_name'] == 'Thailand'].index[0]
    bracket_x = 90

    # Vertical bracket line
    fig.add_shape(
        type='line', xref='x', yref='y',
        x0=bracket_x, x1=bracket_x,
        y0=chile_y, y1=thailand_y,
        line=dict(color=CLUSTER_COLORS['Ready Markets'], width=1.5),
    )
    # Top tick
    fig.add_shape(
        type='line', xref='x', yref='y',
        x0=bracket_x - 1, x1=bracket_x + 1,
        y0=chile_y, y1=chile_y,
        line=dict(color=CLUSTER_COLORS['Ready Markets'], width=1.5),
    )
    # Bottom tick
    fig.add_shape(
        type='line', xref='x', yref='y',
        x0=bracket_x - 1, x1=bracket_x + 1,
        y0=thailand_y, y1=thailand_y,
        line=dict(color=CLUSTER_COLORS['Ready Markets'], width=1.5),
    )
    # Label
    fig.add_annotation(
        x=bracket_x + 0.5,
        y=(chile_y + thailand_y) / 2,
        text=f'<b>~{chile_score - thailand_score:.0f} pt gap</b>',
        showarrow=False, xanchor='left',
        font=dict(size=10, color=CLUSTER_COLORS['Ready Markets'],
                  family='Arial, sans-serif'),
    )

    # Add tier labels on the right side
    tier_positions = {}
    for tier in ['Ready Markets', 'Transition Markets', 'Watch & Wait']:
        tier_indices = sorted_df[sorted_df['cluster_label'] == tier].index.tolist()
        if len(tier_indices) > 0:
            y_pos = np.mean(tier_indices)
            tier_positions[tier] = (y_pos, 94)

    for tier, (y_pos, x_pos) in tier_positions.items():
        fig.add_annotation(
            x=x_pos,
            y=y_pos,
            text=tier,
            showarrow=False,
            xanchor='left',
            font=dict(size=9, color=CLUSTER_COLORS[tier], 
                      family='Arial, sans-serif', weight='bold'),
            bgcolor='rgba(255,255,255,0.85)',
            borderpad=2,
        )

    fig.update_layout(
        **LAYOUT_BASE,
        barmode='overlay',
        height=600,
        margin=dict(l=120, r=110, t=75, b=50),
        xaxis=dict(
            range=[0, 97],
            title='Investment Attractiveness Score (0–100)',
            showgrid=True, gridcolor='#EBEBEB',
            tickfont=dict(size=11), zeroline=False,
        ),
        yaxis=dict(
            tickfont=dict(size=11.5),
            categoryorder='array',
            categoryarray=sorted_df['country_name'].tolist()
        ),
        legend=dict(
            orientation='h', y=-0.1, x=0.5, xanchor='center',
            font=dict(size=11), bgcolor='white',
            bordercolor='#DDDDDD', borderwidth=1,
        ),
    )
    return annotate(fig,
        'Investment Attractiveness Rankings: All 20 Markets',
        'Balanced scenario: Which markets score well overall?',
    )

def chart_governance_scatter(df):
    """Total score vs Business Environment."""
    rdf = df.reset_index()
    fig = go.Figure()

    # Governance threshold band (business environment < 55)
    fig.add_shape(type="rect", x0=0, x1=55, y0=0, y1=100,
                  fillcolor="rgba(176,58,46,0.05)", line=dict(width=0), layer="below")
    fig.add_shape(type="line", x0=55, x1=55, y0=0, y1=100,
                  line=dict(color="#B03A2E", width=1.2, dash="dot"))
    fig.add_annotation(x=27, y=95, text="Governance threshold",
                       showarrow=False, font=dict(size=10, color="#B03A2E"))

    # Trend line with R² calculation
    x_vals = rdf["score_business_environment"].values
    y_vals = rdf["total_score"].values
    m, b = np.polyfit(x_vals, y_vals, 1)
    y_pred = m * x_vals + b
    ss_res = np.sum((y_vals - y_pred) ** 2)
    ss_tot = np.sum((y_vals - np.mean(y_vals)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    x_line = np.linspace(0, 100, 100)
    fig.add_trace(go.Scatter(x=x_line, y=m * x_line + b, mode="lines",
                             name=f"Trend (R² = {r_squared:.2%})", showlegend=True,
                             line=dict(color="#AAAAAA", width=1.5, dash="dash"),
                             hovertemplate="Trend line<extra></extra>"))

    for cluster_lbl, color in CLUSTER_COLORS.items():
        mask = rdf["cluster_label"] == cluster_lbl
        sub  = rdf[mask]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["score_business_environment"],
            y=sub["total_score"],
            mode="markers+text",
            name=cluster_lbl,
            text=sub["country_name"],
            textposition="top center",
            textfont=dict(size=9, color="#444444"),
            marker=dict(size=11, color=color, opacity=0.85,
                        line=dict(width=1.5, color="white")),
            hovertemplate="<b>%{text}</b><br>Governance: %{x:.1f}<br>Total: %{y:.1f}<extra></extra>",
        ))

    fig.update_layout(
        **LAYOUT_BASE,
        height=480,
        margin=dict(l=65, r=40, t=70, b=60),
        xaxis=dict(title="Business Environment Score (0–100)",
                   range=[-2, 107], showgrid=True, gridcolor="#EBEBEB", zeroline=False),
        yaxis=dict(title="Total Attractiveness Score (0–100)",
                   range=[0, 95], showgrid=True, gridcolor="#EBEBEB", zeroline=False),
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center",
                    font=dict(size=11), bgcolor="white",
                    bordercolor="#DDDDDD", borderwidth=1),
    )
    return annotate(fig,
        "Governance Quality Determines Market Attractiveness",
        "Business Environment score (x) vs Total Score (y) · Markets below 55 are not investable",
    )

# ── Remaining chart functions (chart_dotplot, chart_decomposition, chart_sensitivity, 
#    chart_pca, chart_radar) remain unchanged as they are already dynamic ──

def chart_dotplot(df):
    """Dot plot matrix: 4 dimension panels shared y-axis."""
    rdf = df.reset_index().sort_values("total_score", ascending=False).reset_index(drop=True)

    colorscale_map = {
        "score_market_opportunity":
            [[0, "#F0EBE3"], [0.5, "rgba(26,111,168,0.5)"], [1, "#1A6FA8"]],
        "score_decarbonization_opportunity":
            [[0, "#F0EBE3"], [0.5, "rgba(21,25,49,0.5)"],   [1, NAVY]],
        "score_business_environment":
            [[0, "#F0EBE3"], [0.5, "rgba(28,107,69,0.5)"],  [1, "#1C6B45"]],
        "score_energy_security":
            [[0, "#F0EBE3"], [0.5, "rgba(200,145,77,0.5)"], [1, GOLD2]],
    }

    fig = make_subplots(
        rows=1, cols=4, shared_yaxes=True,
        column_titles=[DIM_LABELS[d] for d in DIM_COLS],
        horizontal_spacing=0.03,
    )

    for col_idx, dim in enumerate(DIM_COLS, 1):
        fig.add_trace(go.Scatter(
            x=rdf[dim], y=rdf["country_name"],
            mode="markers",
            marker=dict(
                size=rdf[dim] / 8 + 5,
                color=rdf[dim],
                colorscale=colorscale_map[dim],
                cmin=0, cmax=100,
                showscale=False,
                line=dict(width=0.8, color="white"),
            ),
            text=rdf["country_name"],
            customdata=rdf[dim].round(1),
            hovertemplate="<b>%{text}</b><br>Score: %{customdata}<extra></extra>",
            showlegend=False,
        ), row=1, col=col_idx)

        fig.add_shape(type="line", xref=f"x{col_idx}", yref="paper",
                      x0=50, x1=50, y0=0, y1=1,
                      line=dict(color="#DDDDDD", width=1, dash="dot"))

    # Cluster separators
    watch_n = (rdf["cluster_label"] == "Watch & Wait").sum()
    transition_n = (rdf["cluster_label"] == "Transition Markets").sum()
    for b in [watch_n - 0.5, watch_n + transition_n - 0.5]:
        fig.add_hline(y=b, line=dict(color="#CCCCCC", width=1, dash="dot"))

    fig.update_layout(**LAYOUT_BASE, height=580,
                      margin=dict(l=110, r=40, t=75, b=20))
    fig.update_xaxes(range=[0, 105], showgrid=False,
                     tickvals=[0, 50, 100], tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=10.5), showgrid=True,
                     gridcolor="#F0F0F0", gridwidth=1)
    fig.update_annotations(font=dict(size=11, color=NAVY))
    return annotate(fig,
        "Dimensional Performance: All 20 Markets",
        "Dot size and colour intensity encode score magnitude · Dashed line marks score = 50 · Sorted by total score",
    )

def chart_decomposition(df):
    """Stacked horizontal bar: score decomposition for Malaysia and Chile."""
    DIM_WEIGHTS = [0.30, 0.45, 0.20, 0.05]
    DIM_COLORS_ = ["#1A6FA8", NAVY, "#1C6B45", GOLD2]
    countries = {"Malaysia": "MYS", "Chile": "CHL"}

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12)

    for row_idx, (country_name, code) in enumerate(countries.items(), 1):
        if code not in df.index:
            continue
        row = df.loc[code]
        contrs = [row[k] * w for k, w in zip(DIM_COLS, DIM_WEIGHTS)]
        total = sum(contrs)

        for dim_name, contrib, color in zip(DIM_LABELS.values(), contrs, DIM_COLORS_):
            fig.add_trace(go.Bar(
                name=dim_name, x=[contrib], y=[country_name],
                orientation="h",
                marker=dict(color=color, opacity=0.88),
                text=f"{contrib:.1f}",
                textposition="inside",
                textfont=dict(size=10.5, color="white"),
                hovertemplate=f"<b>{dim_name}</b><br>Contribution: %{{x:.1f}}<extra></extra>",
                showlegend=(row_idx == 1),
                legendgroup=dim_name,
            ), row=row_idx, col=1)

        xref = f"x{row_idx}" if row_idx > 1 else "x"
        yref = f"y{row_idx}" if row_idx > 1 else "y"
        fig.add_annotation(
            xref=xref, yref=yref, x=total + 1.5, y=country_name,
            text=f"<b>{total:.1f}</b>", showarrow=False, xanchor="left",
            font=dict(size=13, color=NAVY),
        )

    fig.update_layout(
        **LAYOUT_BASE, barmode="stack", height=300,
        margin=dict(l=90, r=80, t=70, b=70),
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center",
                    font=dict(size=11), bgcolor="white",
                    bordercolor="#DDDDDD", borderwidth=1, traceorder="normal"),
    )
    fig.update_xaxes(range=[0, 100], showgrid=True, gridcolor="#EBEBEB", tickfont=dict(size=10))
    fig.update_xaxes(title_text="Score Contribution (0–100)", row=2, col=1)
    fig.update_yaxes(showgrid=False, tickfont=dict(size=12))
    return annotate(fig,
        "Score Decomposition: What drives the top 2 rankings?",
        "Each segment shows the weighted contribution of one dimension to the total score",
    )

def chart_sensitivity(df, top_n=14):
    """Dot plot with range lines showing rank spread across scenarios."""
    top = df.nsmallest(top_n, "avg_rank").copy() if "avg_rank" in df.columns else df.head(top_n)
    scenario_cols = ["rank_balanced", "rank_growth_focused", "rank_impact_first", "rank_risk_averse"]
    scenario_cols = [c for c in scenario_cols if c in top.columns]
    scenario_labels = ["Balanced", "Growth-Focused", "Impact-First", "Risk-Averse"]
    scenario_colors = [NAVY, "#1A6FA8", "#1C6B45", GOLD2]
    scenario_symbols = ["circle", "square", "diamond", "cross"]

    countries_ordered = top.sort_values("avg_rank", ascending=False)["country_name"].tolist()

    fig = go.Figure()

    # Range lines (min → max per country)
    for _, mkt in top.iterrows():
        ranks = [mkt[c] for c in scenario_cols if c in top.columns]
        if len(ranks) < 2:
            continue
        fig.add_shape(type="line", xref="x", yref="y",
                      x0=min(ranks), x1=max(ranks),
                      y0=mkt["country_name"], y1=mkt["country_name"],
                      line=dict(color="#CCCCCC", width=2.5), layer="below")

    # Average rank marker
    if "avg_rank" in top.columns:
        fig.add_trace(go.Scatter(
            x=top["avg_rank"], y=top["country_name"],
            mode="markers", name="Average Rank",
            marker=dict(size=7, color="white", symbol="circle",
                        line=dict(width=2, color="#888888")),
            hovertemplate="<b>%{y}</b><br>Avg. Rank: %{x:.1f}<extra></extra>",
        ))

    for col, label, color, sym in zip(scenario_cols, scenario_labels, scenario_colors, scenario_symbols):
        if col not in top.columns:
            continue
        fig.add_trace(go.Scatter(
            x=top[col], y=top["country_name"],
            mode="markers", name=label,
            marker=dict(size=10, color=color, opacity=0.90, symbol=sym,
                        line=dict(width=1, color="white")),
            hovertemplate=f"<b>%{{y}}</b><br>{label}: Rank %{{x}}<extra></extra>",
        ))

    fig.update_layout(
        **LAYOUT_BASE, height=460,
        margin=dict(l=120, r=40, t=70, b=70),
        xaxis=dict(title="Rank Position (lower = better)",
                   autorange="reversed", range=[17, 0.5],
                   tickmode="linear", tick0=1, dtick=1,
                   showgrid=True, gridcolor="#EBEBEB", tickfont=dict(size=10.5)),
        yaxis=dict(categoryorder="array", categoryarray=countries_ordered,
                   tickfont=dict(size=11)),
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center",
                    font=dict(size=11), bgcolor="white",
                    bordercolor="#DDDDDD", borderwidth=1),
    )
    return annotate(fig,
        "Rank Stability Across Four Investor Scenarios",
        "Line span shows min–max rank range · Hollow circle = average rank · Narrower span = more stable",
    )

def chart_pca(df):
    """PCA scatter with convex hull shading per cluster."""
    features = df[DIM_COLS].fillna(50)
    features_scaled = StandardScaler().fit_transform(features)
    pca = PCA(n_components=2, random_state=42).fit(features_scaled)
    coords = pca.transform(features_scaled)
    var = pca.explained_variance_ratio_

    pca_df = df.copy()
    pca_df["pc1"] = coords[:, 0]
    pca_df["pc2"] = coords[:, 1]

    fig = go.Figure()

    for cluster_lbl, color in CLUSTER_COLORS.items():
        mask = pca_df["cluster_label"] == cluster_lbl
        pts = pca_df[mask][["pc1", "pc2"]].values
        if len(pts) < 3:
            continue
        hull = ConvexHull(pts)
        hull_x = list(pts[hull.vertices, 0]) + [pts[hull.vertices[0], 0]]
        hull_y = list(pts[hull.vertices, 1]) + [pts[hull.vertices[0], 1]]
        cx = np.mean(hull_x[:-1])
        cy = np.mean(hull_y[:-1])
        hull_x = [cx + 1.18 * (x - cx) for x in hull_x]
        hull_y = [cy + 1.18 * (y - cy) for y in hull_y]
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        fig.add_trace(go.Scatter(
            x=hull_x, y=hull_y, fill="toself",
            fillcolor=f"rgba({r},{g},{b},0.08)",
            line=dict(color=color, width=1.2, dash="dot"),
            mode="lines", showlegend=False, hoverinfo="skip",
        ))

    for cluster_lbl, color in CLUSTER_COLORS.items():
        mask = pca_df["cluster_label"] == cluster_lbl
        if not mask.any():
            continue
        sub = pca_df[mask]
        fig.add_trace(go.Scatter(
            x=sub["pc1"], y=sub["pc2"],
            mode="markers+text", name=cluster_lbl,
            text=sub["country_name"], textposition="top center",
            textfont=dict(size=8.5, color="#2C3E50"),
            marker=dict(size=sub["total_score"] / 5.5 + 7,
                        color=color, opacity=0.85,
                        line=dict(width=1.5, color="white")),
            hovertemplate="<b>%{text}</b><br>Score: %{customdata:.1f}<extra></extra>",
            customdata=sub["total_score"],
        ))

    fig.update_layout(
        **LAYOUT_BASE, height=480,
        margin=dict(l=55, r=55, t=70, b=80),
        xaxis=dict(title=f"PC1 — {var[0]:.0%} variance explained",
                   showgrid=True, gridcolor="#EBEBEB", zeroline=False, tickfont=dict(size=10)),
        yaxis=dict(title=f"PC2 — {var[1]:.0%} variance explained",
                   showgrid=True, gridcolor="#EBEBEB", zeroline=False, tickfont=dict(size=10)),
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center",
                    font=dict(size=11), bgcolor="white",
                    bordercolor="#DDDDDD", borderwidth=1),
    )
    return annotate(fig,
        "Market Segmentation: Principal Component Analysis",
        "Bubble size proportional to total score · Shaded regions show cluster boundaries",
    )

def chart_radar(df):
    """Radar chart: average dimension scores per archetype."""
    arch = df.groupby("cluster_label").agg({
        **{d: "mean" for d in DIM_COLS},
        "total_score": "count",
    }).round(1)
    arch = arch.rename(columns={"total_score": "n"})

    fig = go.Figure()
    dim_display = ["Market\nOpportunity", "Decarbonisation", "Business\nEnvironment", "Energy\nSecurity"]

    for cluster_lbl, color in CLUSTER_COLORS.items():
        if cluster_lbl not in arch.index:
            continue
        vals = [arch.loc[cluster_lbl, d] for d in DIM_COLS]
        vals_closed = vals + [vals[0]]
        dims_closed = dim_display + [dim_display[0]]
        n = int(arch.loc[cluster_lbl, "n"])
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        fig.add_trace(go.Scatterpolar(
            r=vals_closed, theta=dims_closed, fill="toself",
            name=f"{cluster_lbl} (n={n})",
            line=dict(color=color, width=2),
            fillcolor=f"rgba({r},{g},{b},0.12)",
            hovertemplate=f"<b>{cluster_lbl}</b><br>%{{theta}}: %{{r:.1f}}<extra></extra>",
        ))

    fig.update_layout(
        **LAYOUT_BASE, height=420,
        margin=dict(l=60, r=60, t=70, b=60),
        polar=dict(
            bgcolor="#FAFAFA",
            radialaxis=dict(visible=True, range=[0, 100],
                            tickvals=[25, 50, 75, 100],
                            tickfont=dict(size=9, color="#999999"),
                            gridcolor="#DDDDDD", linecolor="#DDDDDD"),
            angularaxis=dict(tickfont=dict(size=11, color=NAVY),
                             linecolor="#DDDDDD", gridcolor="#DDDDDD"),
        ),
        legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center",
                    font=dict(size=11), bgcolor="white",
                    bordercolor="#DDDDDD", borderwidth=1),
    )
    return annotate(fig,
        "Archetype Dimension Profiles",
        "Average dimension score per cluster (0–100) · Larger area = stronger overall profile",
    )

# ── App init ───────────────────────────────────────────────────────────────

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Renewable Energy Investment Analyzer"

TAB_STYLE = dict(
    padding="14px 28px", fontWeight="500", fontSize="14px",
    border="none", background="transparent", color="#666666",
)
TAB_SEL = dict(
    padding="14px 28px", fontWeight="700", fontSize="14px",
    borderBottom=f"3px solid {GOLD}", background="transparent", color=NAVY,
)

#  Layout 

app.layout = html.Div([

    #  Header 
    html.Div([
        html.Div([
            html.Div([
                html.H1("WHERE TO INVEST IN RENEWABLE ENERGY?",
                        style={"color": "white", "fontSize": "22px",
                               "fontWeight": "700", "margin": "0 0 6px 0",
                               "letterSpacing": "-0.3px"}),
                html.Div([
                    html.Span("MACRO-LEVEL MARKET SCREEN · QUANTITATIVE MARKET PRIORITIZATION FRAMEWORK",
                              style={"color": GOLD, "fontSize": "12px", "fontWeight": "600"}),
                ])
            ], style={"flex": "1"}),
            html.Div([
                html.Div("ANALYSIS PERIOD", style={
                    "color": "rgba(255,255,255,0.5)", "fontSize": "10px",
                    "fontWeight": "700", "letterSpacing": "1.5px", "marginBottom": "2px",
                }),
                html.Div("2018 – 2023  ·  World Bank Open Data",
                         style={"color": "white", "fontSize": "12px"}),
            ], style={"textAlign": "right"}),
        ], style={
            "display": "flex", "justifyContent": "space-between",
            "alignItems": "center", "maxWidth": "1400px",
            "margin": "0 auto", "padding": "0 36px",
        })
    ], style={
        "background": NAVY, "padding": "22px 0",
        "borderBottom": f"4px solid {GOLD}",
    }),

    # ── Main body ────────────────────────────────────────────────────────
    html.Div([
        dcc.Tabs(id="tabs", value="overview", children=[
            dcc.Tab(label="Executive Summary", value="overview",
                    style=TAB_STYLE, selected_style=TAB_SEL),
            dcc.Tab(label="Market Rankings", value="rankings",
                    style=TAB_STYLE, selected_style=TAB_SEL),
            dcc.Tab(label="Sensitivity Analysis", value="sensitivity",
                    style=TAB_STYLE, selected_style=TAB_SEL),
            dcc.Tab(label="Market Archetypes", value="archetypes",
                    style=TAB_STYLE, selected_style=TAB_SEL),
            dcc.Tab(label="Scenario Builder", value="scenarios",
                    style=TAB_STYLE, selected_style=TAB_SEL),
        ], style={
            "borderBottom": "1px solid #EBEBEB",
            "background": "white",
            "marginBottom": "32px",
        }),
        html.Div(id="tab-content"),
    ], style={
        "maxWidth": "1400px", "margin": "0 auto",
        "padding": "32px 36px 60px",
        "minHeight": "calc(100vh - 120px)",
    }),

], style={
    "background": "#F7F7F5",
    "fontFamily": "Arial, Helvetica, sans-serif",
})

# ── Tab renderer ──────────────────────────────────────────────────────────

@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    if tab == "overview":    return tab_overview()
    if tab == "rankings":    return tab_rankings()
    if tab == "sensitivity": return tab_sensitivity()
    if tab == "archetypes":  return tab_archetypes()
    if tab == "scenarios":   return tab_scenarios()

# ── Tab 1: Executive Summary ──────────────────────────────────────────────

def kpi_card(number, headline, implication, action, border_color=NAVY):
    return html.Div([
        html.Div(number, style={
            "fontSize": "32px", "fontWeight": "800", "color": NAVY,
            "borderLeft": f"4px solid {GOLD}", "paddingLeft": "14px",
            "lineHeight": "1", "marginBottom": "12px",
        }),
        html.Div(headline, style={
            "fontSize": "14px", "fontWeight": "700", "color": NAVY,
            "marginBottom": "8px", "lineHeight": "1.4",
        }),
        html.Div(implication, style={
            "fontSize": "12px", "color": "#666666", "fontStyle": "italic",
            "marginBottom": "12px", "lineHeight": "1.5",
        }),
        html.Div(action, style={
            "fontSize": "12px", "fontWeight": "700", "color": GOLD2,
        }),
    ], style={**CARD_STYLE, "borderTop": f"3px solid {GOLD}", "padding": "20px"})

def tab_overview():
    ready_n = (src_df["cluster_label"] == "Ready Markets").sum()
    transition_n = (src_df["cluster_label"] == "Transition Markets").sum()
    watch_n = (src_df["cluster_label"] == "Watch & Wait").sum()
    
    return html.Div([
        # Three KPI cards
        html.Div([
            kpi_card(f"{ready_n} / 20",
                     f"markets qualify as Ready Markets ({ready_n/20*100:.0f}% of the sample)",
                     "Concentration, not diversification. Only Malaysia and Chile score high enough to justify investment today. "
                     f"A further {transition_n} markets require structural improvements.",
                     "Focus on Malaysia & Chile now"),
            kpi_card("49.6 pts",
                     "governance gap between Ready and Transition Markets",
                     "Governance is the primary gatekeeper. Markets below a minimum governance threshold are not investable "
                     "regardless of energy demand or decarbonisation potential.",
                     "Screen for governance first"),
            kpi_card("0.0",
                     "rank standard deviation for Malaysia & Chile across all four scenarios",
                     "The result is robust across all investor profiles (from impact-first to risk-averse).",
                     "No-regret deployment, act immediately"),
        ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr",
                  "gap": "20px", "marginBottom": "24px"}),

        # Strategic implication callout
        html.Div([
            html.Span("Strategic implication:", 
                      style={"fontWeight": "700", "color": NAVY, "marginRight": "8px"}),
            f"Capital should concentrate in the strongest markets rather than being spread across marginal opportunities. "
            f"For the {transition_n} Transition Markets, the best approach is to monitor and wait, with clear conditions "
            f"for re-entering when governance scores improve. The {watch_n} Watch & Wait markets face fundamental barriers "
            f"that make them non-viable on realistic timelines.",
        ], style={
            "background": "#FFF8EC", "border": f"1px solid {GOLD}",
            "borderLeft": f"4px solid {GOLD2}", "borderRadius": "3px",
            "padding": "16px 20px", "fontSize": "14px", "color": "#3D3D3D",
            "marginBottom": "32px", "lineHeight": "1.5",
        }),

        # Rankings chart
        html.Div([
            header_block("Results", "Investment Attractiveness Rankings: All 20 Markets",
                         "Balanced scenario: Which markets score well overall?"),
            dcc.Graph(figure=chart_rankings(src_df), config={"displayModeBar": False}),
        ], style={**CARD_STYLE, "marginBottom": "24px"}),

        # Governance scatter with R²
        html.Div([
            header_block("Core Finding", "The Governance Gate",
                         "Business environment quality separates investable markets from non-investable"),
            dcc.Graph(figure=chart_governance_scatter(src_df), config={"displayModeBar": False}),
        ], style={**CARD_STYLE, "marginBottom": "24px"}),

        # Archetype summary row with profile descriptions
        html.Div([
            html.Div([
                html.Div("Ready Markets", style={
                    "fontSize": "11px", "fontWeight": "700",
                    "color": CLUSTER_COLORS["Ready Markets"],
                    "textTransform": "uppercase", "letterSpacing": "1px",
                    "marginBottom": "4px",
                }),
                html.Div(f"n = {ready_n}  ·  Avg. score: {src_df[src_df['cluster_label']=='Ready Markets']['total_score'].mean():.1f}",
                         style={"fontSize": "13px", "color": NAVY, "fontWeight": "600",
                                "marginBottom": "6px"}),
                html.Div("Malaysia, Chile",
                         style={"fontSize": "12px", "color": "#555555"}),
                html.Div("Strong across all dimensions. Governance scores above 98 place these markets "
                        "in a category of their own.",
                         style={"fontSize": "12px", "color": "#666666", "marginTop": "8px", "lineHeight": "1.4"}),
            ], style={
                **CARD_STYLE, "borderTop": f"4px solid {CLUSTER_COLORS['Ready Markets']}",
                "padding": "18px", "background": "#F0FDF4",
            }),
            html.Div([
                html.Div("Transition Markets", style={
                    "fontSize": "11px", "fontWeight": "700",
                    "color": CLUSTER_COLORS["Transition Markets"],
                    "textTransform": "uppercase", "letterSpacing": "1px",
                    "marginBottom": "4px",
                }),
                html.Div(f"n = {transition_n}  ·  Avg. score: {src_df[src_df['cluster_label']=='Transition Markets']['total_score'].mean():.1f}",
                         style={"fontSize": "13px", "color": NAVY, "fontWeight": "600",
                                "marginBottom": "6px"}),
                html.Div("Thailand, Vietnam, Mexico, Indonesia, Ghana, Senegal, Côte d'Ivoire, "
                        "Bangladesh, Peru, Morocco, Colombia, Philippines, Brazil, Cambodia, Tanzania",
                         style={"fontSize": "12px", "color": "#555555"}),
                html.Div("Strong demand signals but governance gaps require mitigation.",
                         style={"fontSize": "12px", "color": "#666666", "marginTop": "8px", "lineHeight": "1.4"}),
            ], style={
                **CARD_STYLE, "borderTop": f"4px solid {CLUSTER_COLORS['Transition Markets']}",
                "padding": "18px", "background": "#FFFBEB",
            }),
            html.Div([
                html.Div("Watch & Wait", style={
                    "fontSize": "11px", "fontWeight": "700",
                    "color": CLUSTER_COLORS["Watch & Wait"],
                    "textTransform": "uppercase", "letterSpacing": "1px",
                    "marginBottom": "4px",
                }),
                html.Div(f"n = {watch_n}  ·  Avg. score: {src_df[src_df['cluster_label']=='Watch & Wait']['total_score'].mean():.1f}",
                         style={"fontSize": "13px", "color": NAVY, "fontWeight": "600",
                                "marginBottom": "6px"}),
                html.Div("Nigeria, Ethiopia, Kenya",
                         style={"fontSize": "12px", "color": "#555555"}),
                html.Div("Structural barriers prevent near-term viability.",
                         style={"fontSize": "12px", "color": "#666666", "marginTop": "8px", "lineHeight": "1.4"}),
            ], style={
                **CARD_STYLE, "borderTop": f"4px solid {CLUSTER_COLORS['Watch & Wait']}",
                "padding": "18px", "background": "#FEF2F2",
            }),
        ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "20px"}),
    ])

# ── Tab 2: Market Rankings ─────────────────────────────────────────────────

def tab_rankings():
    rdf = src_df.reset_index()
    table_data = rdf[["country_name", "region", "total_score", "rank",
                       "score_market_opportunity",
                       "score_decarbonization_opportunity",
                       "score_business_environment",
                       "score_energy_security",
                       "cluster_label"]].copy()
    table_data.columns = ["Country", "Region", "Total", "Rank",
                          "Market Opp.", "Decarbonisation",
                          "Business Env.", "Energy Sec.", "Archetype"]
    for c in ["Total", "Market Opp.", "Decarbonisation", "Business Env.", "Energy Sec."]:
        table_data[c] = pd.to_numeric(table_data[c], errors="coerce").round(1)
    table_data = table_data.sort_values("Rank")

    return html.Div([
        # Rankings bar chart
        html.Div([
            header_block("Rankings", "Investment Attractiveness Rankings: All 20 Markets",
                         "Balanced scenario · Colour bands indicate archetype"),
            dcc.Graph(id="rankings-chart",
                      figure=chart_rankings(src_df),
                      config={"displayModeBar": False}),
        ], style={**CARD_STYLE, "marginBottom": "24px"}),

        # Dot plot
        html.Div([
            header_block("Dimensions", "Dimensional Performance: All 20 Markets",
                         "Dot size and colour intensity encode score magnitude · Dashed line marks score = 50"),
            dcc.Graph(figure=chart_dotplot(src_df), config={"displayModeBar": False}),
        ], style={**CARD_STYLE, "marginBottom": "24px"}),

        # Score decomposition
        html.Div([
            header_block("Decomposition", "Score Decomposition: What drives the top 2 rankings?",
                         "Each segment shows the weighted contribution of one dimension to the total score"),
            dcc.Graph(figure=chart_decomposition(src_df), config={"displayModeBar": False}),
        ], style={**CARD_STYLE, "marginBottom": "24px"}),

        # Complete data table
        html.Div([
            header_block("Appendix", "Complete Rankings — All 20 Markets with Dimension Scores and Archetype", None),
            dash_table.DataTable(
                data=table_data.to_dict("records"),
                columns=[{"name": c, "id": c, "type": "numeric" if c not in
                           ["Country","Region","Archetype"] else "text"}
                         for c in table_data.columns],
                style_table={"overflowX": "auto"},
                style_header={
                    "backgroundColor": NAVY, "color": "white",
                    "fontWeight": "600", "fontSize": "11px",
                    "padding": "10px 8px",
                    "borderBottom": f"3px solid {GOLD}",
                },
                style_cell={
                    "textAlign": "left", "padding": "9px 8px",
                    "fontSize": "12px", "color": "#3D3D3D",
                    "borderBottom": "1px solid #E5E9F0",
                },
                style_data_conditional=[
                    {"if": {"row_index": "odd"}, "backgroundColor": "#F8F9FA"},
                    {"if": {"column_id": "Archetype",
                            "filter_query": '{Archetype} = "Ready Markets"'},
                     "backgroundColor": "#D4EDDA", "color": "#0D4028", "fontWeight": "600"},
                    {"if": {"column_id": "Archetype",
                            "filter_query": '{Archetype} = "Transition Markets"'},
                     "backgroundColor": "#FFF3CD", "color": "#5A3D00", "fontWeight": "600"},
                    {"if": {"column_id": "Archetype",
                            "filter_query": '{Archetype} = "Watch & Wait"'},
                     "backgroundColor": "#FDECEA", "color": "#7A1F1F", "fontWeight": "600"},
                ],
                page_size=20,
                sort_action="native",
            ),
        ], style=CARD_STYLE),
    ])

# ── Tab 3: Sensitivity Analysis ────────────────────────────────────────────

def tab_sensitivity():
    stab = sensitivity_df.copy()
    if "avg_rank" in stab.columns:
        stab = stab.nsmallest(14, "avg_rank")

    stab_display = stab[["country_name", "region", "avg_rank", "rank_std",
                          "rank_min", "rank_max", "rank_stability"]].copy()
    stab_display.columns = ["Country", "Region", "Avg. Rank", "Std. Dev.",
                             "Best Rank", "Worst Rank", "Stability"]
    stab_display["Avg. Rank"] = stab_display["Avg. Rank"].round(1)
    stab_display["Std. Dev."] = stab_display["Std. Dev."].round(1)

    return html.Div([
        html.Div([
            html.Span("The key question for any scoring exercise is whether results reflect genuine market quality or are an artefact of the chosen weights. ", 
                      style={"color": "#3D3D3D"}),
            html.Span("We answer this by re-running the full model under four investor profiles and comparing each market's rank across scenarios. "
                      "Markets with zero rank variance are high-confidence selections regardless of the specific strategy used.",
                      style={"color": "#666666"}),
        ], style={"fontSize": "14px", "lineHeight": "1.6",
                  "marginBottom": "24px", "padding": "18px 20px",
                  "background": "white", "border": "1px solid #EBEBEB",
                  "borderRadius": "3px"}),

        # Sensitivity dot plot
        html.Div([
            header_block("Sensitivity Analysis", "Rank Stability Across Four Investor Scenarios",
                         "Top 14 markets · Line span = min–max range · Hollow circle = average rank"),
            dcc.Graph(figure=chart_sensitivity(sensitivity_df),
                      config={"displayModeBar": False}),
        ], style={**CARD_STYLE, "marginBottom": "24px"}),

        # Stability insight callout
        html.Div([
            html.Span("The stability insight: ", 
                      style={"fontWeight": "700", "color": NAVY}),
            "Malaysia and Chile hold rank #1 and #2 in every scenario with a standard deviation of zero. "
            "A pension fund (Risk-Averse) and a growth equity firm (Growth-Focused) arrive at the same two "
            "top markets through entirely different analytical lenses, confirming these rankings reflect "
            "genuine market quality, not a modelling choice.",
        ], style={
            "background": "#F0FDF4", "border": "1px solid #1C6B4540",
            "borderLeft": f"4px solid {CLUSTER_COLORS['Ready Markets']}",
            "borderRadius": "3px", "padding": "16px 20px",
            "fontSize": "14px", "color": "#3D3D3D", "marginBottom": "24px", "lineHeight": "1.5",
        }),

        # Stability table
        html.Div([
            header_block("Detail", "Rank Stability Statistics — Top 14 Markets", None),
            dash_table.DataTable(
                data=stab_display.to_dict("records"),
                columns=[{"name": c, "id": c,
                          "type": "numeric" if c not in ["Country","Region","Stability"] else "text"}
                         for c in stab_display.columns],
                style_table={"overflowX": "auto"},
                style_header={
                    "backgroundColor": NAVY, "color": "white", "fontWeight": "600",
                    "fontSize": "11px", "padding": "10px 8px",
                    "borderBottom": f"3px solid {GOLD}",
                },
                style_cell={
                    "textAlign": "left", "padding": "9px 8px",
                    "fontSize": "12px", "color": "#3D3D3D",
                    "borderBottom": "1px solid #E5E9F0",
                },
                style_data_conditional=[
                    {"if": {"row_index": "odd"}, "backgroundColor": "#F8F9FA"},
                    {"if": {"column_id": "Stability",
                            "filter_query": '{Stability} = "Very High"'},
                     "backgroundColor": "#D4EDDA", "color": "#0D4028", "fontWeight": "700"},
                    {"if": {"column_id": "Stability",
                            "filter_query": '{Stability} = "High"'},
                     "backgroundColor": "#FFF3CD", "color": "#5A3D00", "fontWeight": "700"},
                    {"if": {"column_id": "Stability",
                            "filter_query": '{Stability} = "Medium"'},
                     "backgroundColor": "#FDECEA", "color": "#7A1F1F", "fontWeight": "700"},
                ],
                sort_action="native",
            ),
        ], style=CARD_STYLE),
    ])

# ── Tab 4: Market Archetypes ──────────────────────────────────────────────

def tab_archetypes():
    arch = src_df.groupby("cluster_label").agg(
        Market_Opp =("score_market_opportunity", "mean"),
        Decarbonisation=("score_decarbonization_opportunity", "mean"),
        Business_Env =("score_business_environment", "mean"),
        Energy_Sec =("score_energy_security", "mean"),
        Avg_Score =("total_score", "mean"),
        n =("total_score", "count"),
    ).round(1)

    return html.Div([
        # PCA + Radar side by side
        html.Div([
            html.Div([
                header_block("Segmentation", "Market Segmentation: Principal Component Analysis",
                             "Bubble size proportional to total score · Shaded regions show cluster boundaries"),
                dcc.Graph(figure=chart_pca(src_df), config={"displayModeBar": False}),
            ], style={**CARD_STYLE, "flex": "1"}),
            html.Div([
                header_block("Profiles", "Archetype Dimension Profiles",
                             "Average dimension score per cluster (0–100) · Larger area = stronger overall profile"),
                dcc.Graph(figure=chart_radar(src_df), config={"displayModeBar": False}),
            ], style={**CARD_STYLE, "flex": "1"}),
        ], style={"display": "flex", "gap": "20px", "marginBottom": "24px"}),

        # Archetype definition table
        html.Div([
            header_block("Definitions", "Market Archetype Definitions and Recommended Actions", None),
            dash_table.DataTable(
                data=[
                    {"Archetype": "Ready Markets", "N": 2, "Avg. Score": 78.4,
                     "Markets": "Malaysia, Chile",
                     "Profile": "Strong across all dimensions. Governance scores above 98 place these markets in a category of their own."},
                    {"Archetype": "Transition Markets", "N": 15, "Avg. Score": 49.3,
                     "Markets": "Thailand, Vietnam, Mexico, Indonesia, Ghana, Senegal, Côte d'Ivoire, Bangladesh, Peru, Morocco, Colombia, Philippines, Brazil, Cambodia, Tanzania",
                     "Profile": "Strong demand signals and meaningful decarbonisation opportunity. Governance gaps or energy security constraints require mitigation before entry."},
                    {"Archetype": "Watch & Wait", "N": 3, "Avg. Score": 28.4,
                     "Markets": "Nigeria, Ethiopia, Kenya",
                     "Profile": "Structural barriers (governance deficits, sub-critical market size, or infrastructure gaps) prevent near-term viability."},
                ],
                columns=[
                    {"name": "Archetype", "id": "Archetype"},
                    {"name": "N", "id": "N", "type": "numeric"},
                    {"name": "Avg. Score", "id": "Avg. Score", "type": "numeric"},
                    {"name": "Markets", "id": "Markets"},
                    {"name": "Profile", "id": "Profile"},
                ],
                style_table={"overflowX": "auto"},
                style_header={
                    "backgroundColor": NAVY, "color": "white", "fontWeight": "600",
                    "fontSize": "11px", "padding": "10px 8px",
                    "borderBottom": f"3px solid {GOLD}",
                },
                style_cell={
                    "textAlign": "left", "padding": "9px 8px",
                    "fontSize": "11.5px", "color": "#3D3D3D",
                    "borderBottom": "1px solid #E5E9F0",
                },
                style_data_conditional=[
                    {"if": {"row_index": 0}, "backgroundColor": "#D4EDDA", "color": "#0D4028", "fontWeight": "600"},
                    {"if": {"row_index": 1}, "backgroundColor": "#FFF3CD", "color": "#5A3D00", "fontWeight": "600"},
                    {"if": {"row_index": 2}, "backgroundColor": "#FDECEA", "color": "#7A1F1F", "fontWeight": "600"},
                ],
                css=[{"selector": ".dash-cell", "rule": "white-space: normal; line-height: 1.5;"}],
            ),
        ], style=CARD_STYLE),
    ])

# ── Tab 5: Scenario Builder ────────────────────────────────────────────────

def tab_scenarios():
    preset_btn = lambda label, bid, color: html.Button(label, id=bid, n_clicks=0, style={
        "backgroundColor": color, "color": "white", "border": "none",
        "padding": "10px 22px", "borderRadius": "3px", "fontSize": "13px",
        "fontWeight": "600", "cursor": "pointer", "marginRight": "10px",
    })

    def slider_row(label, desc, sid, default):
        return html.Div([
            html.Div([
                html.Span(label, style={"fontWeight": "600", "color": NAVY, "fontSize": "14px"}),
                html.Span(f" — {desc}", style={"color": "#888888", "fontSize": "12px"}),
            ], style={"marginBottom": "8px"}),
            dcc.Slider(id=sid, min=0, max=100, step=5, value=default,
                       marks={i: f"{i}%" for i in range(0, 101, 20)},
                       tooltip={"placement": "bottom", "always_visible": True}),
        ], style={"marginBottom": "28px"})

    return html.Div([
        html.Div([
            header_block("Interactive", "Custom Scenario Builder",
                         "Adjust dimension weights to reflect your investment mandate"),

            # Preset buttons
            html.Div([
                preset_btn("Balanced", "btn-bal", NAVY),
                preset_btn("Impact-First", "btn-impact", "#1C6B45"),
                preset_btn("Growth-Focused", "btn-growth", "#1A6FA8"),
                preset_btn("Risk-Averse", "btn-risk", "#C8914D"),
            ], style={"marginBottom": "28px"}),

            # Sliders
            slider_row("Market Opportunity", "GDP growth, electricity consumption, urbanisation",
                       "sl-mo", 30),
            slider_row("Decarbonisation Opportunity", "Energy use, fossil electricity share, modern renewable gap",
                       "sl-dc", 45),
            slider_row("Business Environment", "Political stability, regulatory quality, rule of law",
                       "sl-be", 20),
            slider_row("Energy Security", "Net energy imports (fixed at 5% across all scenarios)",
                       "sl-es", 5),

            html.Div(id="weight-total", style={
                "padding": "14px", "background": "#F7F7F5",
                "borderRadius": "3px", "fontSize": "14px",
                "fontWeight": "600", "textAlign": "center",
                "border": "1px solid #EBEBEB",
            }),
        ], style={**CARD_STYLE, "marginBottom": "24px"}),

        html.Div([
            header_block("Results", "Custom Rankings",
                         "Recalculated in real time as you adjust weights"),
            dcc.Graph(id="custom-chart", config={"displayModeBar": False}),
        ], style=CARD_STYLE),
    ])

# ── Callbacks: Scenario Builder ────────────────────────────────────────────

@app.callback(
    [Output("sl-mo", "value"), Output("sl-dc", "value"),
     Output("sl-be", "value"), Output("sl-es", "value")],
    [Input("btn-bal", "n_clicks"), Input("btn-impact", "n_clicks"),
     Input("btn-growth", "n_clicks"), Input("btn-risk", "n_clicks")],
    prevent_initial_call=True,
)
def update_sliders(bal, imp, gro, risk):
    btn = callback_context.triggered[0]["prop_id"].split(".")[0]
    presets = {
        "btn-bal":    (30, 45, 20, 5),
        "btn-impact": (20, 60, 15, 5),
        "btn-growth": (55, 25, 15, 5),
        "btn-risk":   (25, 30, 40, 5),
    }
    return presets.get(btn, (30, 45, 20, 5))

@app.callback(
    Output("weight-total", "children"),
    [Input("sl-mo", "value"), Input("sl-dc", "value"),
     Input("sl-be", "value"), Input("sl-es", "value")],
)
def update_total(mo, dc, be, es):
    total = mo + dc + be + es
    if total == 100:
        return html.Span(f"✓ Total weight: {total}% — valid configuration",
                         style={"color": "#1C6B45"})
    return html.Span(f"⚠ Total weight: {total}% — must equal 100%",
                     style={"color": "#B03A2E"})

@app.callback(
    Output("custom-chart", "figure"),
    [Input("sl-mo", "value"), Input("sl-dc", "value"),
     Input("sl-be", "value"), Input("sl-es", "value")],
)
def update_custom(mo, dc, be, es):
    if mo + dc + be + es != 100:
        fig = go.Figure()
        fig.add_annotation(
            text="Adjust weights to total 100% to see updated rankings",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color="#888888"),
        )
        fig.update_layout(**LAYOUT_BASE, height=500)
        return fig
    weights = [mo / 100, dc / 100, be / 100, es / 100]
    return chart_rankings(src_df, custom_weights=weights)

# ── Run ────────────────────────────────────────────────────────────────────
server = app.server

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  WHERE TO INVEST IN RENEWABLE ENERGY?")
    print("  Macro-Level Market Screen — Quantitative Market Prioritization Framework")
    print("=" * 60)
    print(f"\n  Markets loaded: {len(src_df)}")
    print(f"  Archetypes: {dict(src_df['cluster_label'].value_counts())}")
    print("\n  http://localhost:8050")
    print("  Ctrl+C to stop\n")
    app.run(debug=False, host='0.0.0.0', port=8050, dev_tools_silence_routes_logging=True)