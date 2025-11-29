# Macrofiscal Intelligence Dashboard

An interactive **Streamlit application** for analyzing, forecasting, and simulating macroeconomic and fiscal indicators across multiple countries and sources.  
This app transforms raw fiscal data into actionable insights aligned with the **Sustainable Development Goals (SDGs)**.

---

## 📊 Dataset

The app uses a cleaned dataset (`10Alytics_Cleaned_Data.xlsx`) with the following structure:

| Column        | Description                                                                 |
|---------------|-----------------------------------------------------------------------------|
| Country       | Country name (e.g., Egypt, Nigeria, Ghana)                                  |
| Indicator     | Economic/fiscal metric (e.g., Budget Deficit, GDP, Inflation, Health Spend) |
| Source        | Data source (e.g., Central Bank, IMF, World Bank)                           |
| Unit          | Measurement unit (e.g., Million, Percent)                                   |
| Currency      | Currency of measurement (e.g., EGP, NGN, USD)                               |
| Frequency     | Data frequency (Yearly, Quarterly, Monthly)                                 |
| Country Code  | ISO country code                                                            |
| Time          | Original time field (may contain formatting issues, not used in analysis)   |
| Amount        | Numeric value of the indicator                                              |
| Year          | Year of observation                                                         |

---

## 🚀 Features

### Filters
- Select **Country**, **Indicator**, and **Source**.
- Optional filters for **Unit**, **Currency**, and **Frequency**.
- Safe aggregation of duplicate years (`mean`, `sum`, or `median`).

### Tabs
- **Overview**: KPIs, summary statistics, quick snapshot.
- **Trends**: Interactive line chart of selected indicator.
- **Forecasts**: ARIMA-based forecasts with confidence intervals.
- **ML Simulator**: Gradient Boosting model with lag/rolling features; scenario sliders to simulate policy effects.
- **Compare**: Multi-country comparison normalized to index=100 for shape analysis.
- **Export**: Download filtered series and forecasts as CSV.

### Robust Handling
- Drops duplicate years to avoid reindex errors.
- Aggregates multiple entries per year safely.
- Works across multiple indicators and sources.

### SDG Alignment
- Links fiscal and macro trends to SDGs:
  - **SDG 16**: Transparency & institutions
  - **SDG 8**: Growth & employment
  - **SDG 9**: Infrastructure
  - **SDG 10**: Inequality
  - **SDG 1 & 2**: Poverty & hunger
  - **SDG 3 & 4**: Health & education

---

## 🛠 Installation

Clone the repository and install dependencies:

```bash
git clone <your-repo-url>
cd <your-repo>
python -m venv venv
venv\Scripts\activate   # On Windows
pip install -r requirements.txt