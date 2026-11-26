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

from analytics_core import (
    clean_name,
    infer_date_columns,
    infer_numeric_columns,
    infer_categorical_columns,
    detect_business_columns,
    read_workbook,
    clean_dataframe,
    kpi_pack,
    fmt,
    safe_money,
    choose_dimension,
    generate_insights,
    correlation_table,
    anomaly_frame,
    natural_question,
    make_excel_bytes,
    build_pdf,
    chart_html,
    powerbi_assets,
)

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
