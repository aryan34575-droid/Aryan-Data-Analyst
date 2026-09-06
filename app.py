import io
import re
import zipfile
from datetime import datetime
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(
    page_title="Aryan Data Analyst Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Premium UI
# -----------------------------
st.markdown(
    """
<style>
.block-container {padding-top:1.1rem; padding-bottom:3rem; max-width:1500px;}
.hero {padding:1.4rem 1.5rem; border-radius:20px; border:1px solid rgba(120,120,120,.18); background:linear-gradient(135deg, rgba(99,102,241,.12), rgba(14,165,233,.08)); margin-bottom:1rem;}
.hero h1 {margin:0 0 .25rem 0; font-size:2.25rem;}
.hero p {margin:0; opacity:.82;}
.section-card {padding:1rem 1.15rem; border:1px solid rgba(120,120,120,.18); border-radius:16px; margin:.5rem 0 1rem 0;}
.badge {display:inline-block; padding:.25rem .55rem; border-radius:999px; border:1px solid rgba(120,120,120,.25); font-size:.78rem; margin-right:.3rem;}
.small-muted {opacity:.68; font-size:.85rem;}
[data-testid="stMetricValue"] {font-size:1.45rem;}
div[data-testid="stTabs"] button {font-weight:600;}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Core helpers
# -----------------------------
DATE_HINTS = ["date", "time", "month", "year", "day", "timestamp"]

def clean_name(x):
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")


def infer_date_columns(df):
    cols = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            cols.append(c)
            continue
        if any(h in clean_name(c) for h in DATE_HINTS):
            parsed = pd.to_datetime(s, errors="coerce")
            if len(s) and parsed.notna().mean() >= 0.60:
                cols.append(c)
    return cols


def infer_numeric_columns(df):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def infer_categorical_columns(df):
    out = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_datetime64_any_dtype(df[c]):
            continue
        unique = df[c].nunique(dropna=True)
        if unique <= min(100, max(20, int(max(len(df), 1) * 0.20))):
            out.append(c)
    return out


def detect_business_columns(df):
    result = {"sales": None, "profit": None, "quantity": None, "discount": None, "revenue": None, "cost": None}
    aliases = {
        "sales": ["sales", "sale", "net_sales", "sales_amount"],
        "profit": ["profit", "net_profit", "gross_profit"],
        "quantity": ["quantity", "qty", "units", "unit_count"],
        "discount": ["discount", "discount_pct", "discount_percent"],
        "revenue": ["revenue", "turnover", "income"],
        "cost": ["cost", "cogs", "cost_of_goods_sold"],
    }
    for c in df.columns:
        n = clean_name(c)
        for key, words in aliases.items():
            if result[key] is None and any(w == n or w in n for w in words):
                result[key] = c
    if result["sales"] is None:
        result["sales"] = result["revenue"]
    return result


def read_workbook(uploaded):
    raw = uploaded.getvalue()
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return {"CSV": pd.read_csv(io.BytesIO(raw), low_memory=False)}
    if name.endswith((".xlsx", ".xlsm", ".xls")):
        sheets = pd.read_excel(io.BytesIO(raw), sheet_name=None)
        return {k: v for k, v in sheets.items() if isinstance(v, pd.DataFrame)}
    raise ValueError("Supported files: CSV, XLSX, XLSM, XLS")


def clean_dataframe(df):
    out = df.copy()
    audit = []
    before_rows, before_cols = len(out), out.shape[1]

    empty_cols = [c for c in out.columns if out[c].isna().all()]
    if empty_cols:
        out = out.drop(columns=empty_cols)
        audit.append(f"Removed {len(empty_cols):,} completely empty column(s).")

    empty_rows = int(out.isna().all(axis=1).sum())
    if empty_rows:
        out = out.dropna(how="all")
        audit.append(f"Removed {empty_rows:,} completely empty row(s).")

    for c in out.select_dtypes(include="object").columns:
        out[c] = out[c].map(lambda x: x.strip() if isinstance(x, str) else x)

    for c in infer_date_columns(out):
        if not pd.api.types.is_datetime64_any_dtype(out[c]):
            parsed = pd.to_datetime(out[c], errors="coerce")
            if parsed.notna().mean() >= 0.60:
                out[c] = parsed
                audit.append(f"Parsed '{c}' as date/time.")

    for c in list(out.columns):
        if out[c].dtype == "object":
            s = out[c].astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False)
            parsed = pd.to_numeric(s, errors="coerce")
            if len(out[c]) and parsed.notna().mean() >= 0.90 and out[c].nunique(dropna=True) > 5:
                out[c] = parsed
                audit.append(f"Converted '{c}' to numeric.")

    dupes = int(out.duplicated().sum())
    if dupes:
        out = out.drop_duplicates()
        audit.append(f"Removed {dupes:,} exact duplicate row(s).")

    if not audit:
        audit.append("No destructive cleaning was required.")
    audit.append(f"Final shape: {len(out):,} rows × {out.shape[1]:,} columns (from {before_rows:,} × {before_cols:,}).")
    return out, audit


def kpi_pack(df):
    b = detect_business_columns(df)
    pack = {"Rows": int(len(df)), "Columns": int(df.shape[1])}
    if b["sales"]:
        pack["Total Sales"] = float(df[b["sales"]].sum(skipna=True))
        pack["Average Transaction Value"] = float(df[b["sales"]].mean(skipna=True))
    if b["profit"]:
        pack["Total Profit"] = float(df[b["profit"]].sum(skipna=True))
    if b["quantity"]:
        pack["Total Quantity"] = float(df[b["quantity"]].sum(skipna=True))
    if b["sales"] and b["profit"]:
        sales = df[b["sales"]].sum(skipna=True)
        profit = df[b["profit"]].sum(skipna=True)
        pack["Profit Margin %"] = float((profit / sales) * 100) if sales else np.nan
    if b["discount"]:
        pack["Average Discount"] = float(df[b["discount"]].mean(skipna=True))
    return pack


def fmt(k, v):
    if pd.isna(v):
        return "—"
    if k == "Profit Margin %":
        return f"{v:,.2f}%"
    if k == "Average Discount":
        return f"{v:,.2%}" if abs(v) <= 1 else f"{v:,.2f}"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    return f"{v:,.2f}"


def safe_money(v):
    return "—" if pd.isna(v) else f"{v:,.2f}"


def choose_dimension(df):
    cats = infer_categorical_columns(df)
    if not cats:
        return None
    return min(cats, key=lambda c: (df[c].nunique(dropna=True) > 30, df[c].nunique(dropna=True)))


def generate_insights(df):
    b = detect_business_columns(df)
    cats = infer_categorical_columns(df)
    insights = []
    dim = choose_dimension(df)

    if b["sales"] and dim:
        g = df.groupby(dim, dropna=False)[b["sales"]].sum().sort_values(ascending=False)
        if not g.empty:
            total = g.sum()
            share = g.iloc[0] / total * 100 if total else 0
            insights.append(f"**Sales leader:** {g.index[0]} contributes {safe_money(g.iloc[0])} ({share:.1f}% of detected sales).")

    if b["profit"]:
        loss_rows = int((df[b["profit"]] < 0).sum())
        if loss_rows:
            insights.append(f"**Profit risk:** {loss_rows:,} rows have negative profit ({loss_rows/len(df):.1%} of rows).")
        if dim:
            g = df.groupby(dim, dropna=False)[b["profit"]].sum().sort_values()
            if not g.empty:
                insights.append(f"**Lowest-profit segment:** {g.index[0]} at {safe_money(g.iloc[0])}.")

    if b["sales"] and b["profit"]:
        sales = df[b["sales"]].sum()
        profit = df[b["profit"]].sum()
        if sales:
            insights.append(f"**Overall margin:** detected profit margin is {(profit/sales):.2%}.")

    if b["discount"] and b["profit"]:
        pair = df[[b["discount"], b["profit"]]].dropna()
        if len(pair) >= 3:
            corr = pair.corr().iloc[0, 1]
            if pd.notna(corr):
                insights.append(f"**Discount relationship:** correlation with profit is {corr:.2f}; correlation is not causation.")

    dates = infer_date_columns(df)
    if dates and b["sales"]:
        d = dates[0]
        t = df[[d, b["sales"]]].dropna().copy()
        if len(t) >= 12:
            t[d] = pd.to_datetime(t[d], errors="coerce")
            t = t.dropna(subset=[d]).set_index(d)[b["sales"]].resample("ME").sum()
            if len(t) >= 6:
                recent = t.tail(3).mean()
                prior = t.iloc[-6:-3].mean()
                if prior:
                    insights.append(f"**Recent trend:** average monthly sales changed {(recent/prior-1):+.1%} in the latest 3 months vs the prior 3 months.")
    return insights[:12]


def correlation_table(df):
    nums = infer_numeric_columns(df)
    if len(nums) < 2:
        return pd.DataFrame()
    c = df[nums].corr(numeric_only=True)
    pairs = []
    for i, a in enumerate(c.columns):
        for j, b in enumerate(c.columns):
            if j <= i:
                continue
            v = c.loc[a, b]
            if pd.notna(v):
                pairs.append((a, b, float(v), abs(float(v))))
    if not pairs:
        return pd.DataFrame()
    return pd.DataFrame(pairs, columns=["Metric A", "Metric B", "Correlation", "Absolute"]).sort_values("Absolute", ascending=False).drop(columns=["Absolute"]).head(20)


def anomaly_frame(df, date_col, value_col):
    x = df[[date_col, value_col]].dropna().copy()
    if x.empty:
        return x, pd.DataFrame()
    x[date_col] = pd.to_datetime(x[date_col], errors="coerce")
    x = x.dropna(subset=[date_col]).groupby(date_col, as_index=False)[value_col].sum().sort_values(date_col)
    if len(x) < 8:
        return x, pd.DataFrame()
    y = x[value_col].astype(float)
    med = y.rolling(7, min_periods=3, center=True).median()
    resid = y - med
    mad = (resid - resid.median()).abs().median()
    if mad == 0 or pd.isna(mad):
        x["anomaly_score"] = 0.0
        x["anomaly"] = False
        return x, x.iloc[0:0]
    score = (resid - resid.median()).abs() / (1.4826 * mad)
    x["anomaly_score"] = score
    x["anomaly"] = score >= 3.5
    return x, x[x["anomaly"]].copy()


def natural_question(df, q):
    q0 = q.lower().strip()
    b = detect_business_columns(df)
    sales, profit, qty, discount = b["sales"], b["profit"], b["quantity"], b["discount"]

    if any(w in q0 for w in ["total sales", "sales total", "sales kitni", "sales kya hai"]):
        if sales:
            return f"### Total Sales\n**{safe_money(df[sales].sum())}**\n\nCalculation: SUM(`{sales}`) across {len(df):,} rows."
    if any(w in q0 for w in ["total profit", "profit total", "profit kitna"]):
        if profit:
            return f"### Total Profit\n**{safe_money(df[profit].sum())}**\n\nCalculation: SUM(`{profit}`) across {len(df):,} rows."
    if any(w in q0 for w in ["average discount", "avg discount", "discount average"]):
        if discount:
            return f"### Average Discount\n**{df[discount].mean():.2%}**\n\nCalculation: MEAN(`{discount}`)."
    if any(w in q0 for w in ["missing", "null", "blank"]):
        miss = df.isna().sum().sort_values(ascending=False)
        miss = miss[miss > 0]
        if miss.empty:
            return "### Data Quality\nNo missing values detected."
        return "### Missing Values\n\n" + "\n".join(f"- `{c}`: {int(v):,}" for c, v in miss.items())
    if any(w in q0 for w in ["columns", "schema", "fields"]):
        return "### Schema\n\n" + "\n".join(f"- `{c}` — {df[c].dtype}" for c in df.columns)

    metric = profit if "profit" in q0 and profit else sales
    if metric and any(w in q0 for w in ["top", "highest", "best", "largest"]):
        cats = infer_categorical_columns(df)
        chosen = None
        for c in cats:
            tokens = clean_name(c).split("_")
            if any(t in q0 for t in tokens):
                chosen = c
                break
        if chosen is None and cats:
            chosen = choose_dimension(df)
        if chosen:
            n = 10
            m = df.groupby(chosen, dropna=False)[metric].sum().sort_values(ascending=False).head(n)
            label = "Profit" if metric == profit else "Sales"
            return f"### Top {n} {chosen} by {label}\n\n" + "\n".join(f"{i}. **{idx}** — {safe_money(v)}" for i, (idx, v) in enumerate(m.items(), 1))

    return "I can answer common KPI, ranking, schema and data-quality questions locally. Try **total sales**, **total profit**, **top products by sales**, **average discount**, or **show missing values**."


def make_excel_bytes(df, profile, kpis_df, insights_df, corr_df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Cleaned_Data", index=False)
        profile.to_excel(writer, sheet_name="Data_Profile", index=False)
        kpis_df.to_excel(writer, sheet_name="KPIs", index=False)
        insights_df.to_excel(writer, sheet_name="Insights", index=False)
        if not corr_df.empty:
            corr_df.to_excel(writer, sheet_name="Correlations", index=False)
    return buf.getvalue()


def build_pdf(title, sections):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=38, leftMargin=38, topMargin=38, bottomMargin=38)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 12)]
    for heading, body in sections:
        story.append(Paragraph(heading, styles["Heading2"]))
        for line in body:
            story.append(Paragraph(escape(str(line)), styles["BodyText"]))
            story.append(Spacer(1, 4))
        story.append(Spacer(1, 8))
    doc.build(story)
    return buf.getvalue()


def chart_html(fig):
    return fig.to_html(full_html=True, include_plotlyjs=True).encode("utf-8")


def powerbi_assets(df, b):
    measures = []
    if b["sales"]:
        measures.append(f"Total Sales = SUM('Data'[{b['sales']}])")
        measures.append(f"Average Transaction Value = AVERAGE('Data'[{b['sales']}])")
    if b["profit"]:
        measures.append(f"Total Profit = SUM('Data'[{b['profit']}])")
    if b["quantity"]:
        measures.append(f"Total Quantity = SUM('Data'[{b['quantity']}])")
    if b["sales"] and b["profit"]:
        measures.append("Profit Margin % = DIVIDE([Total Profit], [Total Sales], 0)")
    dax = "\n\n".join(measures) if measures else "-- No standard business measures detected."
    pq = """// Aryan Data Analyst Pro - Power Query starter\n// Import the cleaned_data.csv generated by the app.\n\nlet\n    Source = Csv.Document(File.Contents(\"aryan_cleaned_data.csv\"), [Delimiter=\",\", Encoding=65001, QuoteStyle=QuoteStyle.Csv]),\n    PromotedHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars=true])\nin\n    PromotedHeaders\n"""
    blueprint = """POWER BI DASHBOARD BLUEPRINT\n\n1. Executive Overview\n- KPI cards: Sales, Profit, Margin, Quantity\n- Monthly sales/profit trend\n- Top dimension contribution\n- Slicers: Date, Region/Category/Segment where available\n\n2. Sales & Profit\n- Trend line\n- Category/sub-category bars\n- Profitability table\n- Discount vs profit scatter when fields exist\n\n3. Products / Customers\n- Top and bottom performers\n- Contribution and ranking\n- Customer/segment breakdown when identifiers exist\n\n4. Data Quality\n- Missing values\n- Duplicate rows\n- Data types\n- Cleaning actions\n"""
    return dax, pq, blueprint

# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<div class="hero"><h1>📊 Aryan Data Analyst Pro</h1><p>Calculation-first Excel/CSV analytics • dashboards • insights • reports • Power BI-ready exports</p></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 📂 Data Workspace")
    uploaded = st.file_uploader("Upload Excel / CSV", type=["xlsx", "xlsm", "xls", "csv"], key="upload_data_file_v2")
    st.divider()
    st.markdown("## ⚙️ Session")
    if st.button("🔄 Reset analysis session", key="reset_analysis_v2", use_container_width=True):
        st.session_state.clear()
        st.rerun()
    st.caption("Core calculations run locally with pandas. No paid API key is required.")
    st.caption("For confidential business data, use a private deployment you control.")

if not uploaded:
    st.markdown("### Your one-stop analytics workspace")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📊 Analyze", "Excel / CSV")
    c2.metric("🧹 Clean", "Automatic")
    c3.metric("📑 Report", "PDF + Excel")
    c4.metric("⚡ Power BI", "Ready assets")
    st.info("Upload a workbook to unlock the complete workspace. The app is designed so you can use the same deployment for months without rebuilding it.")
    st.markdown("#### Included in this version")
    st.markdown("**Executive dashboard • Data Quality • Trends • Segments • Anomalies • Insights • Ask Aryan • Chart Studio • Power BI • Report Center • Complete Download Pack**")
    st.stop()

try:
    sheets = read_workbook(uploaded)
    if not sheets:
        raise ValueError("No readable worksheet was found.")
except Exception as e:
    st.error(f"Could not read the file: {e}")
    st.stop()

sheet_names = list(sheets.keys())
selected = st.selectbox("Worksheet", sheet_names, key="worksheet_selector_v2")
raw_df = sheets[selected]
df, audit = clean_dataframe(raw_df)
b = detect_business_columns(df)
kpis = kpi_pack(df)
insights = generate_insights(df)
profile = pd.DataFrame({
    "Column": list(df.columns),
    "Type": [str(df[c].dtype) for c in df.columns],
    "Non-null": [int(df[c].notna().sum()) for c in df.columns],
    "Missing": [int(df[c].isna().sum()) for c in df.columns],
    "Missing %": [round(float(df[c].isna().mean()*100), 2) for c in df.columns],
    "Unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
})
kpis_df = pd.DataFrame([{"KPI": k, "Value": v} for k, v in kpis.items()])
insights_df = pd.DataFrame({"Insight": [re.sub(r"\*", "", x) for x in insights]})
corr = correlation_table(df)
dax, pq, blueprint = powerbi_assets(df, b)

st.success(f"Loaded **{selected}** • **{len(df):,} rows × {df.shape[1]:,} columns** • cleaned analysis ready")

# -----------------------------
# Navigation
# -----------------------------
page = st.radio(
    "Workspace",
    ["🏠 Executive", "🔍 Data Quality", "📈 Trends", "🧩 Segments", "🚨 Anomalies", "🧠 Insights", "💬 Ask Aryan", "🎨 Chart Studio", "⚡ Power BI", "📦 Report Center"],
    horizontal=True,
    key="workspace_nav_v2",
)

# -----------------------------
# Executive
# -----------------------------
if page == "🏠 Executive":
    st.subheader("Executive Overview")
    items = list(kpis.items())
    for start in range(0, len(items), 5):
        cols = st.columns(min(5, len(items[start:start+5])))
        for col, (k, v) in zip(cols, items[start:start+5]):
            col.metric(k, fmt(k, v))

    st.markdown("### 📌 Automatic Business Readout")
    if b["sales"] and b["profit"]:
        sales = df[b["sales"]].sum()
        profit = df[b["profit"]].sum()
        margin = profit / sales if sales else 0
        st.markdown(f"Detected **Sales {safe_money(sales)}**, **Profit {safe_money(profit)}**, **Margin {margin:.2%}** across **{len(df):,} rows**.")
    elif b["sales"]:
        st.markdown(f"Detected total sales/revenue of **{safe_money(df[b['sales']].sum())}** across **{len(df):,} rows**.")
    else:
        st.info("No standard sales/revenue field was confidently detected. Use Chart Studio and the raw numeric fields for custom analysis.")

    if insights:
        for x in insights[:5]:
            st.write("• " + x.replace("**", ""))

    dates = infer_date_columns(df)
    if b["sales"] and dates:
        temp = df[[dates[0], b["sales"]]].dropna().copy()
        temp[dates[0]] = pd.to_datetime(temp[dates[0]], errors="coerce")
        temp = temp.dropna(subset=[dates[0]])
        trend = temp.set_index(dates[0])[b["sales"]].resample("ME").sum().reset_index()
        if not trend.empty:
            fig = px.line(trend, x=dates[0], y=b["sales"], markers=True, title="Sales Trend")
            st.plotly_chart(fig, use_container_width=True, key="executive_sales_trend_v2")
    elif b["sales"]:
        dim = choose_dimension(df)
        if dim:
            agg = df.groupby(dim, dropna=False)[b["sales"]].sum().sort_values(ascending=False).head(15).reset_index()
            fig = px.bar(agg, x=dim, y=b["sales"], title=f"Sales by {dim}")
            fig.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig, use_container_width=True, key="executive_sales_bar_v2")

# -----------------------------
# Data Quality
# -----------------------------
elif page == "🔍 Data Quality":
    st.subheader("Data Quality & Cleaning Center")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
    c4.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    st.markdown("### Cleaning Log")
    for x in audit:
        st.write("• " + x)
    st.markdown("### Column Profile")
    st.dataframe(profile, use_container_width=True, hide_index=True)
    st.markdown("### Preview")
    st.dataframe(df.head(100), use_container_width=True, hide_index=True)

# -----------------------------
# Trends
# -----------------------------
elif page == "📈 Trends":
    st.subheader("Trend Analysis")
    dates, nums = infer_date_columns(df), infer_numeric_columns(df)
    if not dates or not nums:
        st.warning("A usable date column and numeric measure are required.")
    else:
        dc = st.selectbox("Date", dates, key="trend_date_v2")
        default_num = nums.index(b["sales"]) if b["sales"] in nums else 0
        vc = st.selectbox("Measure", nums, index=default_num, key="trend_measure_v2")
        freq = st.selectbox("Period", ["ME", "W", "D"], index=0, key="trend_period_v2")
        temp = df[[dc, vc]].dropna().copy()
        temp[dc] = pd.to_datetime(temp[dc], errors="coerce")
        temp = temp.dropna(subset=[dc])
        trend = temp.set_index(dc)[vc].resample(freq).sum().reset_index()
        fig = px.line(trend, x=dc, y=vc, markers=True, title=f"{vc} Trend")
        st.plotly_chart(fig, use_container_width=True, key="trend_chart_v2")
        st.dataframe(trend.tail(100), use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download trend CSV", trend.to_csv(index=False).encode("utf-8"), "trend_analysis.csv", "text/csv", key="download_trend_csv_v2")

# -----------------------------
# Segments
# -----------------------------
elif page == "🧩 Segments":
    st.subheader("Segment / Dimension Analysis")
    cats, nums = infer_categorical_columns(df), infer_numeric_columns(df)
    if not cats or not nums:
        st.warning("Need at least one categorical and one numeric field.")
    else:
        cat = st.selectbox("Dimension", cats, key="segment_dimension_v2")
        default_num = nums.index(b["sales"]) if b["sales"] in nums else 0
        metric = st.selectbox("Metric", nums, index=default_num, key="segment_metric_v2")
        top_n = st.slider("Top N", 5, min(50, max(5, df[cat].nunique(dropna=True))), 10, key="segment_top_n_v2")
        agg = df.groupby(cat, dropna=False)[metric].agg(["sum", "mean", "count"]).sort_values("sum", ascending=False).head(top_n).reset_index()
        fig = px.bar(agg, x=cat, y="sum", title=f"{metric} by {cat}")
        fig.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True, key="segment_chart_v2")
        st.dataframe(agg, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download segment CSV", agg.to_csv(index=False).encode("utf-8"), "segment_analysis.csv", "text/csv", key="download_segment_csv_v2")

# -----------------------------
# Anomalies
# -----------------------------
elif page == "🚨 Anomalies":
    st.subheader("Anomaly Radar")
    dates, nums = infer_date_columns(df), infer_numeric_columns(df)
    if not dates or not nums:
        st.warning("A date and numeric measure are required.")
    else:
        dc = st.selectbox("Date column", dates, key="anom_date_v2")
        default_num = nums.index(b["sales"]) if b["sales"] in nums else 0
        vc = st.selectbox("Measure", nums, index=default_num, key="anom_metric_v2")
        series, anomalies = anomaly_frame(df, dc, vc)
        if series.empty:
            st.warning("Not enough valid observations.")
        else:
            fig = px.line(series, x=dc, y=vc, title=f"{vc} with anomaly flags")
            if not anomalies.empty:
                fig.add_scatter(x=anomalies[dc], y=anomalies[vc], mode="markers", name="Anomaly")
            st.plotly_chart(fig, use_container_width=True, key="anomaly_chart_v2")
            if anomalies.empty:
                st.success("No strong anomalies detected with the conservative robust rule.")
            else:
                st.warning(f"{len(anomalies):,} anomalous period(s) detected.")
                st.dataframe(anomalies, use_container_width=True, hide_index=True)
                st.download_button("⬇️ Download anomalies CSV", anomalies.to_csv(index=False).encode("utf-8"), "anomalies.csv", "text/csv", key="download_anomalies_v2")

# -----------------------------
# Insights
# -----------------------------
elif page == "🧠 Insights":
    st.subheader("Analyst Insights")
    st.caption("Evidence-backed deterministic findings. The app does not invent facts outside the uploaded data.")
    if insights:
        for i, x in enumerate(insights, 1):
            st.markdown(f"**{i}.** {x}")
    else:
        st.info("Not enough recognizable business fields for automatic insights.")
    st.markdown("### Correlation Scan")
    if corr.empty:
        st.info("At least two numeric fields are needed.")
    else:
        st.dataframe(corr, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download correlations CSV", corr.to_csv(index=False).encode("utf-8"), "correlations.csv", "text/csv", key="download_corr_v2")
    st.markdown("### Numeric Summary")
    nums = infer_numeric_columns(df)
    if nums:
        summary = df[nums].describe().T.reset_index().rename(columns={"index": "Metric"})
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download numeric summary", summary.to_csv(index=False).encode("utf-8"), "numeric_summary.csv", "text/csv", key="download_summary_v2")

# -----------------------------
# Ask Aryan
# -----------------------------
elif page == "💬 Ask Aryan":
    st.subheader("Ask Aryan")
    st.caption("Fast local business Q&A for common KPI, ranking, schema and data-quality questions.")
    q = st.text_input("Business question", key="ask_aryan_question_v2", placeholder="e.g. total profit / top products by sales / show missing values")
    if q:
        st.markdown(natural_question(df, q))
    st.markdown("#### Try these")
    examples = ["Total sales", "Total profit", "Top products by sales", "Average discount", "Show missing values", "Show schema"]
    for i, ex in enumerate(examples):
        st.write(f"• {ex}")

# -----------------------------
# Chart Studio
# -----------------------------
elif page == "🎨 Chart Studio":
    st.subheader("Chart Studio")
    nums, cats, dates = infer_numeric_columns(df), infer_categorical_columns(df), infer_date_columns(df)
    chart_type = st.selectbox("Chart type", ["Bar", "Line", "Scatter", "Histogram", "Box", "Pie"], key="chart_type_v2")
    if chart_type == "Bar":
        if not cats or not nums:
            st.warning("Need categorical + numeric fields.")
        else:
            x = st.selectbox("Category", cats, key="chart_bar_x_v2")
            y = st.selectbox("Value", nums, index=nums.index(b["sales"]) if b["sales"] in nums else 0, key="chart_bar_y_v2")
            top = st.slider("Top N", 5, 50, 15, key="chart_bar_top_v2")
            data = df.groupby(x, dropna=False)[y].sum().sort_values(ascending=False).head(top).reset_index()
            fig = px.bar(data, x=x, y=y, title=f"{y} by {x}")
            st.plotly_chart(fig, use_container_width=True, key="chart_bar_plot_v2")
    elif chart_type == "Line":
        if not dates or not nums:
            st.warning("Need date + numeric fields.")
        else:
            x = st.selectbox("Date", dates, key="chart_line_x_v2")
            y = st.selectbox("Value", nums, key="chart_line_y_v2")
            data = df[[x, y]].dropna().copy(); data[x] = pd.to_datetime(data[x], errors="coerce"); data = data.dropna(subset=[x])
            data = data.set_index(x)[y].resample("ME").sum().reset_index()
            fig = px.line(data, x=x, y=y, markers=True, title=f"{y} over time")
            st.plotly_chart(fig, use_container_width=True, key="chart_line_plot_v2")
    elif chart_type == "Scatter":
        if len(nums) < 2:
            st.warning("Need at least two numeric fields.")
        else:
            x = st.selectbox("X", nums, key="chart_scatter_x_v2")
            y = st.selectbox("Y", nums, index=1 if len(nums) > 1 else 0, key="chart_scatter_y_v2")
            fig = px.scatter(df, x=x, y=y, title=f"{y} vs {x}")
            st.plotly_chart(fig, use_container_width=True, key="chart_scatter_plot_v2")
    elif chart_type == "Histogram":
        if not nums:
            st.warning("Need a numeric field.")
        else:
            x = st.selectbox("Measure", nums, key="chart_hist_x_v2")
            fig = px.histogram(df, x=x, title=f"Distribution of {x}")
            st.plotly_chart(fig, use_container_width=True, key="chart_hist_plot_v2")
    elif chart_type == "Box":
        if not nums:
            st.warning("Need a numeric field.")
        else:
            y = st.selectbox("Measure", nums, key="chart_box_y_v2")
            fig = px.box(df, y=y, title=f"Distribution of {y}")
            st.plotly_chart(fig, use_container_width=True, key="chart_box_plot_v2")
    else:
        if not cats or not nums:
            st.warning("Need categorical + numeric fields.")
        else:
            names = st.selectbox("Category", cats, key="chart_pie_names_v2")
            values = st.selectbox("Value", nums, key="chart_pie_values_v2")
            data = df.groupby(names, dropna=False)[values].sum().sort_values(ascending=False).head(12).reset_index()
            fig = px.pie(data, names=names, values=values, title=f"{values} by {names}")
            st.plotly_chart(fig, use_container_width=True, key="chart_pie_plot_v2")
    st.info("Interactive Plotly charts can be explored directly on the page. The complete download pack also includes interactive HTML chart files.")

# -----------------------------
# Power BI
# -----------------------------
elif page == "⚡ Power BI":
    st.subheader("Power BI Ready Center")
    st.info("This creates practical Power BI assets. It does not fabricate a .pbix file; final report assembly happens in Power BI Desktop.")
    st.markdown("### DAX Measures")
    st.code(dax, language="text")
    st.download_button("⬇️ Download DAX", dax, "aryan_DAX_measures.txt", "text/plain", key="download_dax_v2")
    st.markdown("### Power Query Starter")
    st.code(pq, language="text")
    st.download_button("⬇️ Download Power Query", pq, "aryan_powerquery_starter.txt", "text/plain", key="download_pq_v2")
    st.markdown("### Dashboard Blueprint")
    st.code(blueprint, language="text")
    st.download_button("⬇️ Download Dashboard Blueprint", blueprint, "aryan_powerbi_dashboard_blueprint.txt", "text/plain", key="download_blueprint_v2")

# -----------------------------
# Report Center / Everything
# -----------------------------
else:
    st.subheader("📦 Report Center")
    st.markdown("### One-click exports")
    clean_csv = df.to_csv(index=False).encode("utf-8")
    md_lines = [
        "# Aryan Data Analyst Pro — Executive Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Dataset: {uploaded.name} | Worksheet: {selected}",
        "",
        "## KPIs",
    ]
    md_lines += [f"- {k}: {fmt(k, v)}" for k, v in kpis.items()]
    md_lines += ["", "## Cleaning Log"] + [f"- {x}" for x in audit]
    md_lines += ["", "## Insights"] + [f"- {re.sub(r'\*', '', x)}" for x in insights]
    md_lines += ["", "## Power BI"] + blueprint.splitlines()
    report_md = "\n".join(md_lines)
    pdf = build_pdf(
        "Aryan Data Analyst Pro — Executive Report",
        [
            ("Dataset", [f"File: {uploaded.name}", f"Worksheet: {selected}", f"Rows: {len(df):,}", f"Columns: {df.shape[1]:,}"]),
            ("KPIs", [f"{k}: {fmt(k, v)}" for k, v in kpis.items()]),
            ("Data Quality", audit),
            ("Business Insights", [re.sub(r"\*", "", x) for x in insights] or ["No automatic business insights were generated."]),
            ("Power BI", blueprint.splitlines()),
        ],
    )
    excel = make_excel_bytes(df, profile, kpis_df, insights_df, corr)

    # Interactive charts for the download pack
    chart_files = {}
    dates = infer_date_columns(df)
    if b["sales"] and dates:
        t = df[[dates[0], b["sales"]]].dropna().copy(); t[dates[0]] = pd.to_datetime(t[dates[0]], errors="coerce"); t = t.dropna(subset=[dates[0]])
        t = t.set_index(dates[0])[b["sales"]].resample("ME").sum().reset_index()
        chart_files["sales_trend.html"] = chart_html(px.line(t, x=dates[0], y=b["sales"], markers=True, title="Sales Trend"))
    dim = choose_dimension(df)
    if b["sales"] and dim:
        t = df.groupby(dim, dropna=False)[b["sales"]].sum().sort_values(ascending=False).head(15).reset_index()
        chart_files["sales_by_dimension.html"] = chart_html(px.bar(t, x=dim, y=b["sales"], title=f"Sales by {dim}"))
    if b["profit"] and dim:
        t = df.groupby(dim, dropna=False)[b["profit"]].sum().sort_values(ascending=False).head(15).reset_index()
        chart_files["profit_by_dimension.html"] = chart_html(px.bar(t, x=dim, y=b["profit"], title=f"Profit by {dim}"))

    pack_buf = io.BytesIO()
    with zipfile.ZipFile(pack_buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("Aryan_Executive_Report.pdf", pdf)
        z.writestr("Aryan_Executive_Report.md", report_md.encode("utf-8"))
        z.writestr("Aryan_Cleaned_Data.csv", clean_csv)
        z.writestr("Aryan_Analytics_Workbook.xlsx", excel)
        z.writestr("Aryan_Data_Profile.csv", profile.to_csv(index=False).encode("utf-8"))
        z.writestr("Aryan_KPIs.csv", kpis_df.to_csv(index=False).encode("utf-8"))
        z.writestr("Aryan_Insights.csv", insights_df.to_csv(index=False).encode("utf-8"))
        if not corr.empty:
            z.writestr("Aryan_Correlations.csv", corr.to_csv(index=False).encode("utf-8"))
        z.writestr("Aryan_DAX_Measures.txt", dax.encode("utf-8"))
        z.writestr("Aryan_Power_Query.txt", pq.encode("utf-8"))
        z.writestr("Aryan_PowerBI_Dashboard_Blueprint.txt", blueprint.encode("utf-8"))
        z.writestr("README.txt", "Generated by Aryan Data Analyst Pro. Core calculations are deterministic pandas analysis.\n".encode("utf-8"))
        for name, data in chart_files.items():
            z.writestr("Charts/" + name, data)
    complete_pack = pack_buf.getvalue()

    st.success("Complete analysis package is ready.")
    st.download_button("🚀 DOWNLOAD EVERYTHING — ZIP", complete_pack, "Aryan_Data_Analyst_COMPLETE_PACK.zip", "application/zip", key="download_everything_v2", use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.download_button("📊 Cleaned Excel", excel, "Aryan_Analytics_Workbook.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="download_excel_v2")
    c2.download_button("📄 Executive PDF", pdf, "Aryan_Executive_Report.pdf", "application/pdf", key="download_pdf_v2")
    c3.download_button("📝 Markdown Report", report_md, "Aryan_Executive_Report.md", "text/markdown", key="download_md_v2")

    st.markdown("### Individual downloads")
    st.download_button("⬇️ Cleaned CSV", clean_csv, "Aryan_Cleaned_Data.csv", "text/csv", key="download_clean_csv_v2")
    st.download_button("⬇️ DAX Measures", dax, "Aryan_DAX_Measures.txt", "text/plain", key="download_dax_report_v2")
    st.download_button("⬇️ Power Query", pq, "Aryan_Power_Query.txt", "text/plain", key="download_pq_report_v2")
    st.download_button("⬇️ Dashboard Blueprint", blueprint, "Aryan_PowerBI_Dashboard_Blueprint.txt", "text/plain", key="download_blueprint_report_v2")
    st.markdown("### What is inside the ZIP?")
    st.write("PDF report • Markdown report • Cleaned CSV • Excel analytics workbook • KPI/profile/insight exports • Correlations • DAX • Power Query • Power BI blueprint • Interactive HTML charts")

st.caption("Aryan Data Analyst Pro • Stable calculation-first architecture • No paid API key required for core analysis")
