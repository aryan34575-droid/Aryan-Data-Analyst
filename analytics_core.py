"""Reusable, testable analytics functions for Aryan Data Analyst Pro."""

import io
import re
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

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
    """Find unambiguous numeric business measures without mistaking dimensions for metrics."""
    result = {
        "sales": None, "profit": None, "quantity": None, "discount": None,
        "revenue": None, "cost": None, "order_id": None,
    }
    measure_aliases = {
        "sales": {"sale", "sales", "net_sales", "gross_sales", "total_sales", "sales_amount", "sales_value"},
        "profit": {"profit", "net_profit", "gross_profit", "operating_profit", "total_profit"},
        "quantity": {"quantity", "qty", "units", "unit_count", "total_quantity"},
        "discount": {"discount", "discount_pct", "discount_percent", "discount_rate"},
        "revenue": {"revenue", "total_revenue", "net_revenue", "turnover"},
        "cost": {"cost", "total_cost", "cogs", "cost_of_goods_sold"},
    }
    order_aliases = {
        "order", "order_id", "order_number", "sales_order_id", "transaction_id",
        "transaction_number", "invoice_id", "invoice_number", "purchase_id",
    }

    for column in df.columns:
        normalized = clean_name(column)
        if result["order_id"] is None and normalized in order_aliases:
            result["order_id"] = column
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue
        for key, aliases in measure_aliases.items():
            if result[key] is None and normalized in aliases:
                result[key] = column

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
        total_sales = df[b["sales"]].sum(skipna=True)
        pack["Total Sales"] = float(total_sales)
        if b["order_id"]:
            order_count = int(df[b["order_id"]].nunique(dropna=True))
            pack["Distinct Orders"] = order_count
            pack["Average Order Value"] = float(total_sales / order_count) if order_count else np.nan
        else:
            # Do not label a row-level mean as a transaction/order average.
            pack["Average Sales per Row"] = float(df[b["sales"]].mean(skipna=True))
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
    deviation = (resid - resid.median()).abs()
    if pd.isna(mad):
        x["anomaly_score"] = np.nan
        x["anomaly"] = False
        return x, x.iloc[0:0]
    if mad == 0:
        # A zero MAD is common in flat series. Flag non-zero residual deviations
        # rather than silently suppressing an obvious isolated spike.
        x["anomaly_score"] = deviation
        x["anomaly"] = deviation > 0
    else:
        score = deviation / (1.4826 * mad)
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
