# Aryan Data Analyst

A calculation-first, open-source Streamlit data analyst for Excel/CSV work.

## Core features
- XLSX/XLSM/XLS/CSV upload
- Multi-sheet workbook picker
- Conservative cleaning and visible audit
- Automatic KPI detection
- Executive overview
- Trend analysis
- Segment/dimension analysis
- Robust anomaly detection
- Deterministic natural-language questions
- Cleaned CSV export
- Power BI DAX starter measures
- Power Query starter
- Executive Markdown report

## No API key required
The core analysis is deterministic and runs with pandas/NumPy/Plotly. No paid AI API is required.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
This project is designed for Streamlit Community Cloud. Put the files in a GitHub repository and deploy `app.py`.

## Important
This app intentionally does not pretend to generate a `.pbix` file. It generates the cleaned dataset, DAX measures, Power Query starter and dashboard specification needed to finish the Power BI report in Power BI Desktop.

## License
MIT-style use is permitted for this generated project; adapt the files as needed.
