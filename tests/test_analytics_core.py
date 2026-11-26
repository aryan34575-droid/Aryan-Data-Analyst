import io

import numpy as np
import pandas as pd

from analytics_core import (
    anomaly_frame,
    build_pdf,
    clean_dataframe,
    correlation_table,
    detect_business_columns,
    generate_insights,
    kpi_pack,
    make_excel_bytes,
    natural_question,
    powerbi_assets,
)


def sample_sales_data():
    rows = [
        {"Order Date": "2026-01-01", "Sales": "100", "Profit": "10", "Quantity": 1, "Category": " A "},
        {"Order Date": "2026-01-02", "Sales": "200", "Profit": "20", "Quantity": 2, "Category": "B"},
        {"Order Date": "2026-01-03", "Sales": "300", "Profit": "30", "Quantity": 3, "Category": "A"},
        {"Order Date": "2026-01-04", "Sales": "400", "Profit": "40", "Quantity": 4, "Category": "B"},
        {"Order Date": "2026-01-05", "Sales": "500", "Profit": "50", "Quantity": 5, "Category": "A"},
        {"Order Date": "2026-01-06", "Sales": "600", "Profit": "60", "Quantity": 6, "Category": "B"},
    ]
    rows.extend([rows[0].copy(), {key: None for key in rows[0]}])
    for row in rows:
        row["Empty"] = None
    return pd.DataFrame(rows)


def test_clean_dataframe_removes_empty_rows_columns_and_exact_duplicates():
    cleaned, audit = clean_dataframe(sample_sales_data())

    assert cleaned.shape == (6, 5)
    assert "Empty" not in cleaned.columns
    assert cleaned["Sales"].dtype.kind in "if"
    assert pd.api.types.is_datetime64_any_dtype(cleaned["Order Date"])
    assert cleaned.loc[0, "Category"] == "A"
    assert cleaned.duplicated().sum() == 0
    assert any("Removed 1 completely empty row" in item for item in audit)
    assert any("Removed 1 exact duplicate row" in item for item in audit)


def test_kpi_pack_returns_reconciled_sales_profit_and_margin():
    cleaned, _ = clean_dataframe(sample_sales_data())
    kpis = kpi_pack(cleaned)

    assert kpis["Rows"] == 6
    assert kpis["Total Sales"] == 2100.0
    assert kpis["Average Transaction Value"] == 350.0
    assert kpis["Total Profit"] == 210.0
    assert kpis["Total Quantity"] == 21.0
    assert kpis["Profit Margin %"] == 10.0


def test_business_column_detection_avoids_dimension_and_margin_false_matches():
    frame = pd.DataFrame({
        "Sales Region": ["North", "South"],
        "Profit Margin": [0.10, 0.15],
        "Net Sales": [100.0, 200.0],
        "Operating Profit": [10.0, 30.0],
        "Order ID": ["A-1", "A-2"],
    })

    detected = detect_business_columns(frame)

    assert detected["sales"] == "Net Sales"
    assert detected["profit"] == "Operating Profit"
    assert detected["order_id"] == "Order ID"
    assert detected["revenue"] is None
    assert detected["cost"] is None


def test_average_order_value_uses_distinct_orders_not_row_average():
    frame = pd.DataFrame({
        "Order ID": ["A-1", "A-1", "A-2"],
        "Sales": [100.0, 50.0, 150.0],
    })

    kpis = kpi_pack(frame)

    assert kpis["Total Sales"] == 300.0
    assert kpis["Distinct Orders"] == 2
    assert kpis["Average Order Value"] == 150.0
    assert "Average Transaction Value" not in kpis


def test_average_sales_per_row_is_labelled_without_order_identifier():
    frame = pd.DataFrame({"Sales": [100.0, 200.0, 300.0]})

    kpis = kpi_pack(frame)

    assert kpis["Average Sales per Row"] == 200.0
    assert "Average Order Value" not in kpis


def test_anomaly_frame_flags_spike_when_mad_is_zero():
    frame = pd.DataFrame({
        "Date": pd.date_range("2026-02-01", periods=12, freq="D"),
        "Sales": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 100],
    })

    series, anomalies = anomaly_frame(frame, "Date", "Sales")

    assert len(series) == 12
    assert len(anomalies) >= 1
    assert bool(anomalies.iloc[-1]["anomaly"])


def test_business_insights_are_based_on_detected_columns():
    cleaned, _ = clean_dataframe(sample_sales_data())
    columns = detect_business_columns(cleaned)
    insights = generate_insights(cleaned)

    assert columns["sales"] == "Sales"
    assert columns["profit"] == "Profit"
    assert any("Sales leader:** B" in item for item in insights)
    assert any("Overall margin" in item for item in insights)


def test_natural_question_reports_total_sales_and_missing_values():
    cleaned, _ = clean_dataframe(sample_sales_data())

    answer = natural_question(cleaned, "total sales")
    assert "2,100.00" in answer

    with_missing = cleaned.copy()
    with_missing.loc[0, "Profit"] = np.nan
    missing_answer = natural_question(with_missing, "show missing values")
    assert "Profit" in missing_answer


def test_correlation_table_returns_numeric_pairs():
    cleaned, _ = clean_dataframe(sample_sales_data())
    result = correlation_table(cleaned)

    assert not result.empty
    assert list(result.columns) == ["Metric A", "Metric B", "Correlation"]


def test_anomaly_frame_flags_a_large_spike_for_review():
    frame = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=12, freq="D"),
        "Sales": [10, 11, 9, 12, 8, 11, 10, 9, 12, 8, 11, 100],
    })
    series, anomalies = anomaly_frame(frame, "Date", "Sales")

    assert len(series) == 12
    assert len(anomalies) >= 1
    assert bool(anomalies.iloc[-1]["anomaly"])


def test_export_helpers_create_valid_file_signatures():
    cleaned, _ = clean_dataframe(sample_sales_data())
    kpis = kpi_pack(cleaned)
    profile = pd.DataFrame({
        "Column": cleaned.columns,
        "Type": [str(cleaned[c].dtype) for c in cleaned.columns],
    })
    kpis_df = pd.DataFrame([{"KPI": key, "Value": value} for key, value in kpis.items()])
    insights_df = pd.DataFrame({"Insight": generate_insights(cleaned)})
    corr = correlation_table(cleaned)

    workbook = make_excel_bytes(cleaned, profile, kpis_df, insights_df, corr)
    pdf = build_pdf("Analytics Test Report", [("Summary", ["Test report generated."])])

    assert workbook[:2] == b"PK"
    assert pdf.startswith(b"%PDF")


def test_power_bi_assets_uses_detected_business_fields():
    cleaned, _ = clean_dataframe(sample_sales_data())
    dax, power_query, blueprint = powerbi_assets(cleaned, detect_business_columns(cleaned))

    assert "Total Sales = SUM('Data'[Sales])" in dax
    assert "Total Profit = SUM('Data'[Profit])" in dax
    assert "PromotedHeaders" in power_query
    assert "Executive Overview" in blueprint
