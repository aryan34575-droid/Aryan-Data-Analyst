# Aryan Data Analyst Pro

A calculation-first Streamlit analytics workspace for Excel/CSV files.

## Features
- Excel/CSV upload and multi-sheet selection
- Conservative automatic cleaning
- Executive KPI dashboard
- Data quality/profile analysis
- Trend and segment analysis
- Robust anomaly review with a zero-MAD fallback for flat time series
- Evidence-backed deterministic business insights
- Local common-question analyst (Ask Aryan)
- Interactive Chart Studio
- Power BI DAX, Power Query and dashboard blueprint exports
- PDF, Markdown, Excel and CSV exports
- One-click complete ZIP analysis pack with interactive HTML charts
- No paid API key required for core analysis

## Technical structure
- `app.py`: Streamlit presentation and interaction layer
- `analytics_core.py`: reusable data preparation, KPI, insights, anomaly and export functions
- `tests/`: automated unit tests for core analytical behaviour, business-column detection and KPI grain semantics

The KPI engine distinguishes order-level average value from row-level average sales when an order identifier is available. Business measures are detected using conservative numeric-column aliases to reduce false matches. The anomaly detector includes a fallback for flat series where median absolute deviation is zero.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Run tests
```bash
pip install -r requirements-dev.txt
pytest -q
```

GitHub Actions runs the test suite on pushes and pull requests to `main`.

## Deployment
Designed for Streamlit Community Cloud. `runtime.txt` pins Python 3.12.10.

## Important
This is a deterministic analytics application, not a general-purpose LLM agent. It prioritizes reproducible calculations over invented answers. It does not generate a `.pbix` file; it prepares assets for Power BI Desktop.
