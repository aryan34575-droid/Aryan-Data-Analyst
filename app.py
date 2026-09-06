
import io
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(
    page_title="Aryan Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
[data-testid="stMetricValue"] {font-size: 1.55rem;}
.small-muted {color:#777;font-size:.88rem;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Helpers
# -----------------------------
NUMERIC_HINTS = [
    "sales","revenue","amount","price","cost","profit","quantity","qty",
    "discount","margin","income","expense","value","score","rate","total"
]
DATE_HINTS = ["date","time","month","year","day"]

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
            if parsed.notna().mean() >= 0.60:
                cols.append(c)
    return cols

def infer_numeric_columns(df):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

def infer_categorical_columns(df):
    return [
        c for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_datetime64_any_dtype(df[c])
        and df[c].nunique(dropna=True) <= min(100, max(20, int(len(df)*0.20)))
    ]

def score_business_columns(df):
    result = {"sales": None, "profit": None, "quantity": None, "discount": None, "revenue": None}
    for c in df.columns:
        n = clean_name(c)
        for key in list(result):
            if result[key] is None:
                if key == "sales" and "sales" in n:
                    result[key] = c
                elif key == "profit" and "profit" in n:
                    result[key] = c
                elif key == "quantity" and ("quantity" in n or n in ("qty","units")):
                    result[key] = c
                elif key == "discount" and "discount" in n:
                    result[key] = c
                elif key == "revenue" and ("revenue" in n or "turnover" in n):
                    result[key] = c
    if result["sales"] is None and result["revenue"] is not None:
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
    before = len(out)

    # Remove completely empty rows/columns
    empty_cols = [c for c in out.columns if out[c].isna().all()]
    if empty_cols:
        out = out.drop(columns=empty_cols)
        audit.append(f"Removed {len(empty_cols)} completely empty column(s).")

    empty_rows = out.isna().all(axis=1).sum()
    if empty_rows:
        out = out.dropna(how="all")
        audit.append(f"Removed {int(empty_rows):,} completely empty row(s).")

    # Trim text
    for c in out.select_dtypes(include="object").columns:
        out[c] = out[c].map(lambda x: x.strip() if isinstance(x, str) else x)

    # Date inference
    for c in infer_date_columns(out):
        if not pd.api.types.is_datetime64_any_dtype(out[c]):
            parsed = pd.to_datetime(out[c], errors="coerce")
            if parsed.notna().mean() >= 0.60:
                out[c] = parsed
                audit.append(f"Parsed '{c}' as date/time.")

    # Numeric-looking object columns
    for c in list(out.columns):
        if out[c].dtype == "object":
            s = out[c].astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False)
            parsed = pd.to_numeric(s, errors="coerce")
            if parsed.notna().mean() >= 0.90 and out[c].nunique(dropna=True) > 5:
                out[c] = parsed
                audit.append(f"Converted '{c}' to numeric.")

    dupes = int(out.duplicated().sum())
    if dupes:
        out = out.drop_duplicates()
        audit.append(f"Removed {dupes:,} exact duplicate row(s).")

    if len(out) != before:
        audit.append(f"Rows changed from {before:,} to {len(out):,}.")
    if not audit:
        audit.append("No destructive cleaning was required.")

    return out, audit

def kpi_pack(df):
    business = score_business_columns(df)
    numeric = infer_numeric_columns(df)
    pack = {}
    sales = business["sales"]
    profit = business["profit"]
    qty = business["quantity"]
    discount = business["discount"]

    if sales:
        pack["Total Sales"] = float(df[sales].sum(skipna=True))
    if profit:
        pack["Total Profit"] = float(df[profit].sum(skipna=True))
    if qty:
        pack["Total Quantity"] = float(df[qty].sum(skipna=True))
    if sales and profit:
        s = df[sales].sum(skipna=True)
        p = df[profit].sum(skipna=True)
        pack["Profit Margin %"] = float((p / s) * 100) if s else np.nan
    if discount:
        pack["Average Discount"] = float(df[discount].mean(skipna=True))
    if sales:
        pack["Average Transaction Value"] = float(df[sales].mean(skipna=True))
    pack["Rows"] = int(len(df))
    pack["Columns"] = int(df.shape[1])
    return pack

def anomaly_frame(df, date_col, value_col):
    x = df[[date_col, value_col]].dropna().copy()
    if x.empty:
        return x, pd.DataFrame()
    x = x.groupby(date_col, as_index=False)[value_col].sum()
    x = x.sort_values(date_col)
    if len(x) < 8:
        return x, pd.DataFrame()
    y = x[value_col].astype(float)
    med = y.rolling(7, min_periods=3, center=True).median()
    resid = y - med
    mad = (resid - resid.median()).abs().median()
    if mad == 0 or pd.isna(mad):
        x["anomaly"] = False
        return x, x.iloc[0:0]
    score = (resid - resid.median()).abs() / (1.4826 * mad)
    x["anomaly_score"] = score
    x["anomaly"] = score >= 3.5
    return x, x[x["anomaly"]].copy()

def safe_money(x):
    if pd.isna(x):
        return "—"
    return f"{x:,.2f}"

def format_kpi(k, v):
    if k in ("Profit Margin %",):
        return f"{v:,.2f}%"
    if k == "Average Discount":
        return f"{v:,.2%}" if abs(v) <= 1 else f"{v:,.2f}"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    return f"{v:,.2f}"

def natural_question(df, q):
    """Deterministic natural-language analyst for common business questions."""
    q0 = q.lower().strip()
    business = score_business_columns(df)
    sales, profit, qty = business["sales"], business["profit"], business["quantity"]

    # totals
    if any(w in q0 for w in ["total sales", "sales total", "sales kitni", "sales kya hai"]):
        if sales:
            v = df[sales].sum()
            return f"Total Sales = {safe_money(v)}\n\nCalculation: SUM(`{sales}`) over {len(df):,} rows."
    if any(w in q0 for w in ["total profit", "profit total", "profit kitna"]):
        if profit:
            v = df[profit].sum()
            return f"Total Profit = {safe_money(v)}\n\nCalculation: SUM(`{profit}`) over {len(df):,} rows."
    if any(w in q0 for w in ["average discount", "avg discount", "discount average"]):
        c = business["discount"]
        if c:
            v = df[c].mean()
            return f"Average Discount = {v:.2%}\n\nCalculation: MEAN(`{c}`)."

    # top / bottom by numeric business metric
    metric = profit if ("profit" in q0 and profit) else sales
    if metric and any(w in q0 for w in ["top", "highest", "best", "largest"]):
        cats = infer_categorical_columns(df)
        chosen = None
        for c in cats:
            if any(token in q0 for token in clean_name(c).split("_")):
                chosen = c
                break
        if chosen is None:
            for c in cats:
                if df[c].nunique(dropna=True) <= 30:
                    chosen = c
                    break
        if chosen:
            n = 10
            m = df.groupby(chosen, dropna=False)[metric].sum().sort_values(ascending=False).head(n)
            label = "Profit" if metric == profit else "Sales"
            return f"Top {n} {chosen} by {label}:\n\n" + "\n".join(
                [f"{i+1}. {idx}: {safe_money(val)}" for i, (idx, val) in enumerate(m.items())]
            )

    if any(w in q0 for w in ["columns", "schema", "fields"]):
        return "Columns:\n\n" + "\n".join([f"- {c} ({df[c].dtype})" for c in df.columns])

    if any(w in q0 for w in ["missing", "null", "blank"]):
        miss = df.isna().sum().sort_values(ascending=False)
        miss = miss[miss > 0]
        if miss.empty:
            return "No missing values were detected."
        return "Missing values:\n\n" + "\n".join([f"- {c}: {int(v):,}" for c, v in miss.items()])

    return (
        "I can answer common KPI, ranking, schema and data-quality questions locally. "
        "Try: 'total sales', 'total profit', 'top products by sales', "
        "'average discount', 'show missing values', or use the dashboard tabs."
    )


def generate_business_insights(df):
    """Generate deterministic, evidence-backed insights from the detected schema."""
    insights = []
    b = score_business_columns(df)
    sales, profit, qty, disc = b["sales"], b["profit"], b["quantity"], b["discount"]
    cats = infer_categorical_columns(df)

    if sales:
        total_sales = df[sales].sum()
        if cats:
            cat = min(cats, key=lambda c: (df[c].nunique(dropna=True) > 30, df[c].nunique(dropna=True)))
            g = df.groupby(cat, dropna=False)[sales].sum().sort_values(ascending=False)
            if len(g):
                share = (g.iloc[0] / total_sales * 100) if total_sales else 0
                insights.append(
                    f"{cat} '{g.index[0]}' is the largest sales contributor at "
                    f"{safe_money(g.iloc[0])} ({share:.1f}% of detected sales)."
                )

    if profit:
        if cats:
            cat = min(cats, key=lambda c: (df[c].nunique(dropna=True) > 30, df[c].nunique(dropna=True)))
            g = df.groupby(cat, dropna=False)[profit].sum().sort_values()
            if len(g):
                insights.append(
                    f"The weakest detected {cat} by total profit is '{g.index[0]}' "
                    f"with {safe_money(g.iloc[0])}."
                )
        loss_rows = int((df[profit] < 0).sum())
        if loss_rows:
            insights.append(
                f"{loss_rows:,} rows have negative profit ({loss_rows/len(df):.1%} of rows)."
            )

    if sales and profit:
        s, p = df[sales].sum(), df[profit].sum()
        if s:
            margin = p / s
            insights.append(f"Overall detected profit margin is {margin:.2%}.")

    if disc and profit:
        corr = df[[disc, profit]].dropna().corr().iloc[0,1]
        if pd.notna(corr):
            direction = "positive" if corr > 0 else "negative"
            insights.append(
                f"Discount vs profit correlation is {corr:.2f} ({direction}); "
                "this is correlation, not proof of causation."
            )

    dates = infer_date_columns(df)
    if dates and sales:
        d = dates[0]
        t = df[[d, sales]].dropna().copy()
        if len(t) >= 12:
            t[d] = pd.to_datetime(t[d], errors="coerce")
            t = t.dropna(subset=[d]).set_index(d)[sales].resample("M").sum()
            if len(t) >= 6:
                recent = t.tail(3).mean()
                prior = t.iloc[-6:-3].mean()
                if prior:
                    change = (recent/prior - 1) * 100
                    insights.append(
                        f"Average monthly sales changed {change:+.1f}% in the latest 3 months "
                        "versus the preceding 3 months."
                    )
    return insights[:12]

def correlation_table(df):
    numeric = infer_numeric_columns(df)
    if len(numeric) < 2:
        return pd.DataFrame()
    c = df[numeric].corr(numeric_only=True)
    pairs = []
    for i, a in enumerate(c.columns):
        for j, b in enumerate(c.columns):
            if j <= i:
                continue
            val = c.loc[a,b]
            if pd.notna(val):
                pairs.append((a,b,val,abs(val)))
    return pd.DataFrame(pairs, columns=["Metric A","Metric B","Correlation","Absolute"]).sort_values(
        "Absolute", ascending=False
    ).drop(columns=["Absolute"]).head(20)

def build_pdf_report(title, lines):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 12)]
    for line in lines:
        if line.startswith("## "):
            story.append(Paragraph(line[3:], styles["Heading2"]))
        elif line.startswith("- "):
            story.append(Paragraph("• " + line[2:], styles["BodyText"]))
        else:
            story.append(Paragraph(line.replace("&","&amp;"), styles["BodyText"]))
        story.append(Spacer(1, 6))
    doc.build(story)
    return buf.getvalue()

# -----------------------------
# App
# -----------------------------
st.title("📊 Aryan Data Analyst")
st.caption("Autonomous, calculation-first Excel/CSV analysis — built for business reporting and Power BI preparation.")

with st.sidebar:
    st.header("1. Upload data")
    uploaded = st.file_uploader("Excel/CSV", type=["xlsx","xlsm","xls","csv"])
    st.divider()
    st.header("2. Analysis controls")
    st.info("Upload a workbook. ADA-style analysis runs locally with pandas; no API key is required.")
    st.caption("Your uploaded file is processed in the app session. Do not upload confidential data to a public deployment unless you trust the hosting setup.")

if not uploaded:
    st.markdown("""
### What Aryan Data Analyst does

**Upload one Excel workbook** and the app automatically:

- audits data quality and duplicates
- detects dates, measures and business dimensions
- calculates KPIs
- analyzes trends, products, customers and categories
- finds concentration and anomalies
- creates interactive charts
- produces an executive report
- exports cleaned data
- generates a Power BI-ready starter pack with DAX + Power Query guidance

**No API key is required for the core calculations.**
""")
    st.stop()

try:
    sheets = read_workbook(uploaded)
except Exception as e:
    st.error(f"Could not read the file: {e}")
    st.stop()

sheet_names = list(sheets)
selected = st.selectbox("Worksheet", sheet_names)
raw_df = sheets[selected]
df, audit = clean_dataframe(raw_df)

st.success(f"Loaded **{selected}** — {len(df):,} rows × {df.shape[1]:,} columns")

tabs = st.tabs([
    "🏠 Executive",
    "🔍 Data Quality",
    "📈 Trends",
    "🧩 Segments",
    "🚨 Anomalies",
    "🧠 Insights",
    "💬 Ask Aryan",
    "📦 Power BI Pack",
    "📄 Report"
])

business = score_business_columns(df)
kpis = kpi_pack(df)

with tabs[0]:
    st.subheader("Executive overview")
    cols = st.columns(5)
    display = list(kpis.items())[:5]
    for col, (k, v) in zip(cols, display):
        col.metric(k, format_kpi(k, v))

    if len(kpis) > 5:
        cols2 = st.columns(5)
        for col, (k, v) in zip(cols2, list(kpis.items())[5:10]):
            col.metric(k, format_kpi(k, v))

    st.markdown("### Automatic business readout")
    if business["sales"] and business["profit"]:
        sales = df[business["sales"]].sum()
        profit = df[business["profit"]].sum()
        margin = (profit / sales * 100) if sales else 0
        st.write(
            f"The dataset contains **{len(df):,} rows**. "
            f"Detected Sales = **{safe_money(sales)}**, Profit = **{safe_money(profit)}**, "
            f"and Profit Margin = **{margin:.2f}%**."
        )

    cats = infer_categorical_columns(df)
    if business["sales"] and cats:
        chosen = next((c for c in cats if df[c].nunique(dropna=True) <= 20), cats[0])
        agg = df.groupby(chosen, dropna=False)[business["sales"]].sum().sort_values(ascending=False).head(15).reset_index()
        fig = px.bar(agg, x=chosen, y=business["sales"], title=f"Sales by {chosen}")
        fig.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.subheader("Data quality audit")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
    c4.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")

    st.markdown("#### Cleaning log")
    for item in audit:
        st.write("• " + item)

    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if not missing.empty:
        st.dataframe(missing.rename("Missing values").to_frame(), use_container_width=True)
    else:
        st.success("No missing values detected.")

    st.markdown("#### Column profile")
    profile = pd.DataFrame({
        "Column": df.columns,
        "Type": [str(df[c].dtype) for c in df.columns],
        "Non-null": [int(df[c].notna().sum()) for c in df.columns],
        "Missing": [int(df[c].isna().sum()) for c in df.columns],
        "Unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    st.dataframe(profile, use_container_width=True)

with tabs[2]:
    st.subheader("Trend analysis")
    dates = infer_date_columns(df)
    numeric = infer_numeric_columns(df)
    if not dates or not numeric:
        st.warning("A usable date column and numeric measure were not detected.")
    else:
        dc = st.selectbox("Date", dates)
        vc = st.selectbox("Measure", numeric, index=numeric.index(business["sales"]) if business["sales"] in numeric else 0)
        freq = st.selectbox("Period", ["M","W","D"], index=0)
        temp = df[[dc,vc]].dropna().copy()
        temp[dc] = pd.to_datetime(temp[dc], errors="coerce")
        temp = temp.dropna(subset=[dc])
        trend = temp.set_index(dc)[vc].resample(freq).sum().reset_index()
        fig = px.line(trend, x=dc, y=vc, markers=True, title=f"{vc} trend")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(trend.tail(20), use_container_width=True)

with tabs[3]:
    st.subheader("Segment / dimension analysis")
    cats = infer_categorical_columns(df)
    numeric = infer_numeric_columns(df)
    if not cats or not numeric:
        st.warning("Not enough categorical + numeric fields detected.")
    else:
        cat = st.selectbox("Dimension", cats)
        metric = st.selectbox("Metric", numeric, index=numeric.index(business["sales"]) if business["sales"] in numeric else 0)
        top_n = st.slider("Top N", 5, 30, 10)
        agg = df.groupby(cat, dropna=False)[metric].agg(["sum","mean","count"]).sort_values("sum", ascending=False).head(top_n).reset_index()
        fig = px.bar(agg, x=cat, y="sum", title=f"{metric} by {cat}")
        fig.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(agg, use_container_width=True)

with tabs[4]:
    st.subheader("Anomaly radar")
    dates = infer_date_columns(df)
    numeric = infer_numeric_columns(df)
    if not dates or not numeric:
        st.warning("A date and numeric measure are required.")
    else:
        dc = st.selectbox("Date column", dates, key="anom_date")
        vc = st.selectbox("Measure", numeric, index=numeric.index(business["sales"]) if business["sales"] in numeric else 0, key="anom_metric")
        series, anomalies = anomaly_frame(df, dc, vc)
        if series.empty:
            st.warning("Not enough valid observations.")
        else:
            fig = px.line(series, x=dc, y=vc, title=f"{vc} with anomaly flags")
            if not anomalies.empty:
                fig.add_scatter(x=anomalies[dc], y=anomalies[vc], mode="markers", name="Anomaly")
            st.plotly_chart(fig, use_container_width=True)
            if anomalies.empty:
                st.success("No strong anomalies detected with the conservative robust rule.")
            else:
                st.warning(f"{len(anomalies):,} anomalous period(s) detected.")
                st.dataframe(anomalies, use_container_width=True)

with tabs[5]:
    st.subheader("🧠 Analyst Insights")
    st.caption("Evidence-backed deterministic findings generated from the uploaded data.")
    insights = generate_business_insights(df)
    if insights:
        for i, item in enumerate(insights, 1):
            st.write(f"**{i}.** {item}")
    else:
        st.info("Not enough business fields were detected for automatic insight generation.")

    st.markdown("### Correlation scan")
    corr = correlation_table(df)
    if corr.empty:
        st.info("At least two numeric fields are needed.")
    else:
        st.dataframe(corr, use_container_width=True)

    st.markdown("### Numeric summary")
    numeric = infer_numeric_columns(df)
    if numeric:
        st.dataframe(df[numeric].describe().T, use_container_width=True)

with tabs[6]:
    st.subheader("Ask Aryan")
    q = st.text_input("Ask a business question", placeholder="e.g. top products by sales / total profit / show missing values")
    if q:
        answer = natural_question(df, q)
        st.markdown(answer)


    q = st.text_input("Ask a business question", placeholder="e.g. top products by sales / total profit / show missing values")
    if q:
        answer = natural_question(df, q)
        st.markdown(answer)

with tabs[7]:
    st.subheader("Power BI starter pack")
    st.write("This generates practical assets for Power BI Desktop. It does **not** fabricate a .pbix file.")
    clean_csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download cleaned CSV", clean_csv, file_name="aryan_cleaned_data.csv", mime="text/csv")

    measures = []
    if business["sales"]:
        measures.append(f"Total Sales = SUM('Data'[{business['sales']}])")
    if business["profit"]:
        measures.append(f"Total Profit = SUM('Data'[{business['profit']}])")
    if business["quantity"]:
        measures.append(f"Total Quantity = SUM('Data'[{business['quantity']}])")
    if business["sales"] and business["profit"]:
        measures.append("Profit Margin % = DIVIDE([Total Profit], [Total Sales], 0)")
    if business["sales"]:
        measures.append("Average Transaction Value = AVERAGE('Data'[" + business["sales"] + "])")

    dax = "\n\n".join(measures) if measures else "-- No standard business measures were detected."
    st.code(dax, language="text")
    st.download_button("⬇️ Download DAX measures", dax, file_name="aryan_powerbi_measures.txt", mime="text/plain")

    pq = f"""// Aryan Data Analyst - Power Query starter
// Import the cleaned CSV exported by this app.
// Then set correct data types for dates, numeric measures and dimensions.
// Recommended model: one fact table named Data + separate Date table + dimensions where appropriate.

let
    Source = Csv.Document(File.Contents("aryan_cleaned_data.csv"), [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]),
    PromotedHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars=true])
in
    PromotedHeaders
"""
    st.download_button("⬇️ Download Power Query starter", pq, file_name="aryan_powerquery_starter.txt", mime="text/plain")

    st.markdown("### Recommended dashboard pages")
    pages = [
        ("1. Executive Overview", "KPI cards, sales/profit trend, regional/category performance, slicers."),
        ("2. Sales & Profit", "Monthly trend, category/sub-category, discount vs profit, waterfall/drivers."),
        ("3. Customers", "Customer count, top customers, segment performance, repeat/retention analysis when IDs support it."),
        ("4. Products", "Top/bottom products, quantity, sales, profit, discount and contribution."),
        ("5. Operations / Returns", "Returns and shipping analysis when those fields exist."),
    ]
    for name, desc in pages:
        st.write(f"**{name}** — {desc}")

with tabs[8]:
    st.subheader("Executive report")
    sales = business["sales"]
    profit = business["profit"]
    report = []
    report.append("# Aryan Data Analyst — Executive Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append(f"\nDataset: {uploaded.name} | Worksheet: {selected}")
    report.append("\n## Data quality")
    report.extend([f"- {x}" for x in audit])
    report.append("\n## KPIs")
    for k,v in kpis.items():
        report.append(f"- **{k}:** {format_kpi(k,v)}")
    report.append("\n## Business insights")
    if sales:
        top = df.groupby(infer_categorical_columns(df)[0], dropna=False)[sales].sum().sort_values(ascending=False).head(5) if infer_categorical_columns(df) else pd.Series()
        if not top.empty:
            report.append(f"- Top dimension by sales: **{top.index[0]}** with **{safe_money(top.iloc[0])}**.")
    if sales and profit:
        loss = df.groupby(infer_categorical_columns(df)[0], dropna=False)[profit].sum().sort_values().head(5) if infer_categorical_columns(df) else pd.Series()
        if not loss.empty:
            report.append(f"- Lowest-profit dimension: **{loss.index[0]}** with **{safe_money(loss.iloc[0])}**.")
    report.append("\n## Recommended Power BI pages")
    for name, desc in pages:
        report.append(f"- **{name}:** {desc}")
    report_text = "\n".join(report)
    st.download_button("⬇️ Download executive Markdown report", report_text, file_name="aryan_executive_report.md", mime="text/markdown")
    pdf_lines = report_text.split("\n")
    pdf_bytes = build_pdf_report("Aryan Data Analyst — Executive Report", pdf_lines)
    st.download_button("⬇️ Download executive PDF report", pdf_bytes, file_name="aryan_executive_report.pdf", mime="application/pdf")
    st.markdown(report_text)
