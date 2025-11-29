import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score

# --------------------------------
# App configuration
# --------------------------------
st.set_page_config(page_title="Macrofiscal Intelligence | Multi-country, Multi-indicator", layout="wide")
DATA_PATH = Path("data/10Alytics_Cleaned_Data.xlsx")

PRIMARY = "#275d98"
ACCENT = "#d62728"
SECONDARY = "#1f77b4"

# --------------------------------
# Utilities
# --------------------------------
@st.cache_data
def load_data(path: Path):
    if not path.exists():
        raise FileNotFoundError("Excel file not found at data/10Alytics_Cleaned_Data.xlsx")
    df = pd.read_excel(path)

    # Required columns check
    required = ["Country", "Indicator", "Source", "Amount", "Year"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Parse core fields
    df["Year"] = pd.to_datetime(df["Year"], format="%Y", errors="coerce")
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

    # Optional metadata
    for opt in ["Unit", "Currency", "Frequency", "Country Code", "Time"]:
        if opt not in df.columns:
            df[opt] = np.nan

    # Clean "Time" if present but not usable
    # We rely on Year; Time can be ignored if formatting issues exist.
    df = df.dropna(subset=["Year", "Amount"])
    df = df.sort_values(["Country", "Indicator", "Source", "Year"]).reset_index(drop=True)
    return df

def apply_filters(df, country, indicator, source, unit_opt, currency_opt, freq_opt):
    sub = df[
        (df["Country"] == country) &
        (df["Indicator"] == indicator) &
        (df["Source"] == source)
    ].copy()
    if unit_opt and unit_opt != "(any)":
        sub = sub[sub["Unit"] == unit_opt]
    if currency_opt and currency_opt != "(any)":
        sub = sub[sub["Currency"] == currency_opt]
    if freq_opt and freq_opt != "(any)":
        sub = sub[sub["Frequency"] == freq_opt]
    return sub

def aggregate_series(df_slice, agg_method="mean"):
    """Aggregate duplicates by Year safely."""
    if df_slice.empty:
        return pd.DataFrame(columns=["Year", "Value"])
    g = df_slice.groupby("Year")["Amount"]
    if agg_method == "sum":
        agg = g.sum()
    elif agg_method == "median":
        agg = g.median()
    else:
        agg = g.mean()
    s = agg.asfreq("Y")
    s = s.interpolate("time")
    tidy = s.reset_index().rename(columns={"Amount": "Value"})
    return tidy

def arima_auto_order(series):
    """Simple auto selection among a small grid for yearly data."""
    s = series.dropna()
    if len(s) < 8:
        return None
    candidates = [(1,1,1), (2,1,1), (1,1,2), (2,1,2), (0,1,1)]
    best_aic, best_order = np.inf, None
    for order in candidates:
        try:
            model = sm.tsa.arima.ARIMA(s, order=order).fit()
            if model.aic < best_aic:
                best_aic, best_order = model.aic, order
        except:
            continue
    return best_order or (1,1,1)

def arima_forecast(series, steps=5):
    s = series.dropna()
    if len(s) < 8:
        return None, "Insufficient history (< 8 observations) for ARIMA."
    order = arima_auto_order(s) or (1,1,1)
    try:
        model = sm.tsa.arima.ARIMA(s, order=order)
        res = model.fit()
        fc = res.get_forecast(steps=steps).summary_frame()
        return fc, None
    except Exception as e:
        return None, f"Forecast error: {e}"

def build_features(tidy_df, target_col="Value"):
    d = tidy_df.copy()
    # AR lags
    for L in [1,2,3]:
        d[f"{target_col}_lag{L}"] = d[target_col].shift(L)
    # Rolling stats
    d["roll_mean_3"] = d[target_col].rolling(3).mean()
    d["roll_std_3"] = d[target_col].rolling(3).std()
    d["roll_mean_5"] = d[target_col].rolling(5).mean()
    d["roll_std_5"] = d[target_col].rolling(5).std()
    # YoY change (%)
    d["yoy_pct"] = d[target_col].pct_change() * 100
    # Calendar numeric
    d["year_num"] = d["Year"].dt.year
    d = d.dropna().reset_index(drop=True)
    if d.empty:
        return None, None, None, None
    X = d[[c for c in d.columns if c not in ["Year", target_col]]]
    y = d[target_col]
    return d, X, y, X.columns.tolist()

def train_gbr_model(X, y):
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=42
    )
    # Time-aware CV
    if len(X) >= 12:
        tscv = TimeSeriesSplit(n_splits=3)
        cv_mae = -cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_absolute_error")
        cv_r2 = cross_val_score(model, X, y, cv=tscv, scoring="r2")
    else:
        cv_mae, cv_r2 = np.array([]), np.array([])
    model.fit(X, y)
    return model, cv_mae, cv_r2

def amt_fmt(val):
    try:
        return f"{float(val):,.0f}"
    except:
        return "—"

def pct_fmt(val):
    try:
        return f"{float(val):.2f}%"
    except:
        return "—"

def index_to_100(df, value_col, base_year=None):
    d = df.copy()
    if base_year is None:
        base_year = int(d["Year"].min().year)
    base_val = d.loc[d["Year"].dt.year == base_year, value_col]
    if len(base_val) == 0 or base_val.iloc[0] == 0 or pd.isna(base_val.iloc[0]):
        return None
    d["Index"] = d[value_col] / base_val.iloc[0] * 100
    return d, base_year

# --------------------------------
# Load data
# --------------------------------
try:
    data = load_data(DATA_PATH)
except Exception as e:
    st.error(str(e))
    st.stop()

# --------------------------------
# Sidebar filters
# --------------------------------
st.sidebar.header("Filters")
countries = sorted(data["Country"].dropna().unique().tolist())
indicators = sorted(data["Indicator"].dropna().unique().tolist())
sources = sorted(data["Source"].dropna().unique().tolist())
units = sorted(data["Unit"].dropna().unique().tolist()) if "Unit" in data.columns else []
currs = sorted(data["Currency"].dropna().unique().tolist()) if "Currency" in data.columns else []
freqs = sorted(data["Frequency"].dropna().unique().tolist()) if "Frequency" in data.columns else []

selected_country = st.sidebar.selectbox("Country", countries, index=0 if countries else None)
selected_indicator = st.sidebar.selectbox("Indicator", indicators, index=0 if indicators else None)
selected_source = st.sidebar.selectbox("Source", sources, index=0 if sources else None)

selected_unit = st.sidebar.selectbox("Unit (optional)", ["(any)"] + units, index=0) if units else "(any)"
selected_currency = st.sidebar.selectbox("Currency (optional)", ["(any)"] + currs, index=0) if currs else "(any)"
selected_freq = st.sidebar.selectbox("Frequency (optional)", ["(any)"] + freqs, index=0) if freqs else "(any)"

st.sidebar.markdown("---")
agg_method = st.sidebar.radio("Aggregation for duplicate years", ["mean", "sum", "median"], index=0)
compare_countries = st.sidebar.multiselect("Compare countries (same Indicator/Source)", countries[:6], default=[])

# --------------------------------
# Header and meta badges
# --------------------------------
st.title("Macrofiscal Intelligence Dashboard")
st.caption("Explore multi-country, multi-indicator fiscal & macroeconomic series; forecast trends and simulate scenarios.")

meta_cols = st.columns(4)
meta_cols[0].markdown(f"**Country:** {selected_country}")
meta_cols[1].markdown(f"**Indicator:** {selected_indicator}")
meta_cols[2].markdown(f"**Source:** {selected_source}")
meta_cols[3].markdown(f"**Aggregation:** {agg_method}")

# --------------------------------
# Apply filters (updated)
# --------------------------------
filtered = data[
    (data["Country"] == selected_country) &
    (data["Indicator"] == selected_indicator) &
    (data["Source"] == selected_source)
].copy()

# Only apply optional filters if explicitly chosen
if "Unit" in data.columns and selected_unit and selected_unit != "(any)":
    filtered = filtered[filtered["Unit"] == selected_unit]
if "Currency" in data.columns and selected_currency and selected_currency != "(any)":
    filtered = filtered[filtered["Currency"] == selected_currency]
if "Frequency" in data.columns and selected_freq and selected_freq != "(any)":
    filtered = filtered[filtered["Frequency"] == selected_freq]

# Drop duplicate years to avoid reindex errors
filtered = filtered.drop_duplicates(subset=["Year"])

# Debug info (optional, remove later)
st.write("Filtered rows:", len(filtered))
if not filtered.empty:
    st.dataframe(filtered.head())
# Aggregate series
tidy = aggregate_series(filtered, agg_method=agg_method)

# --------------------------------
# Tabs layout
# --------------------------------
tab_overview, tab_trends, tab_forecast, tab_ml, tab_compare, tab_export = st.tabs([
    "Overview", "Trends", "Forecasts", "ML simulator", "Compare", "Export"
])

# --------------------------------
# Overview tab
# --------------------------------
with tab_overview:
    if tidy.empty:
        st.warning("No data for the selected filters. Try changing Unit/Currency/Frequency or aggregation.")
    else:
        latest = tidy.iloc[-1]
        earliest = tidy.iloc[0]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Latest value", amt_fmt(latest["Value"]))
        k2.metric("Earliest value", amt_fmt(earliest["Value"]))
        change = latest["Value"] - earliest["Value"]
        k3.metric("Change since start", amt_fmt(change))
        k4.metric("Observations", f"{len(tidy):,}")

        st.subheader("Summary statistics")
        st.dataframe(tidy["Value"].describe().to_frame().T)

        st.markdown("Use the tabs to explore trends, build forecasts, simulate scenarios, compare countries, and export results.")

# --------------------------------
# Trends tab
# --------------------------------
with tab_trends:
    if tidy.empty:
        st.info("No series to display.")
    else:
        st.subheader(f"{selected_indicator} over time")
        trend_chart = alt.Chart(tidy).mark_line(point=True, color=PRIMARY).encode(
            x=alt.X("Year:T", title="Year"),
            y=alt.Y("Value:Q", title=f"{selected_indicator}"),
            tooltip=["Year", "Value"]
        ).properties(height=380)
        st.altair_chart(trend_chart, use_container_width=True)

# --------------------------------
# Forecasts tab (ARIMA with CI band)
# --------------------------------
with tab_forecast:
    if tidy.empty:
        st.info("No data to forecast.")
    else:
        st.subheader("ARIMA forecast")
        steps = st.slider("Forecast horizon (years)", 3, 10, 5)

        series = tidy.set_index("Year")["Value"]
        fc_frame, fc_err = arima_forecast(series, steps=steps)

        if fc_err:
            st.info(fc_err)
        else:
            last_year = series.index[-1].year
            future_years = pd.date_range(f"{last_year+1}-12-31", periods=steps, freq="Y")

            fc_df = fc_frame.reset_index(drop=True)
            fc_df.insert(0, "Year", future_years)
            fc_df = fc_df.rename(columns={"mean": "Forecast", "mean_ci_lower": "CI Lower", "mean_ci_upper": "CI Upper"})

            st.dataframe(fc_df.assign(
                Forecast=lambda d: d["Forecast"].round(0).astype(int),
                **{"CI Lower": fc_df["CI Lower"].round(0).astype(int),
                   "CI Upper": fc_df["CI Upper"].round(0).astype(int)}
            ))

            hist_df = tidy.rename(columns={"Value": "Actual"})
            fc_plot_df = fc_df[["Year", "Forecast", "CI Lower", "CI Upper"]].copy()

            hist_line = alt.Chart(hist_df).mark_line(color=ACCENT).encode(
                x=alt.X("Year:T", title="Year"),
                y=alt.Y("Actual:Q", title=f"{selected_indicator}")
            )
            fc_line = alt.Chart(fc_plot_df).mark_line(color=SECONDARY).encode(
                x="Year:T", y="Forecast:Q"
            )
            fc_band = alt.Chart(fc_plot_df).mark_area(color=SECONDARY, opacity=0.2).encode(
                x="Year:T", y="CI Upper:Q", y2="CI Lower:Q"
            )
            st.altair_chart((hist_line + fc_band + fc_line).properties(height=380), use_container_width=True)

# --------------------------------
# ML simulator tab (Gradient Boosting)
# --------------------------------
with tab_ml:
    if tidy.empty or len(tidy) < 10:
        st.info("Insufficient data for ML training (need ≥ 10 observations).")
    else:
        st.subheader("Machine learning simulator (next-year projection)")
        # Feature engineering
        feat_df = tidy.copy()
        for L in [1,2,3]:
            feat_df[f"Value_lag{L}"] = feat_df["Value"].shift(L)
        feat_df["roll_mean_3"] = feat_df["Value"].rolling(3).mean()
        feat_df["roll_std_3"] = feat_df["Value"].rolling(3).std()
        feat_df["roll_mean_5"] = feat_df["Value"].rolling(5).mean()
        feat_df["roll_std_5"] = feat_df["Value"].rolling(5).std()
        feat_df["yoy_pct"] = feat_df["Value"].pct_change() * 100
        feat_df["year_num"] = feat_df["Year"].dt.year
        feat_df = feat_df.dropna().reset_index(drop=True)

        if len(feat_df) < 8:
            st.info("Not enough engineered samples for ML training.")
        else:
            X = feat_df[[c for c in feat_df.columns if c not in ["Year", "Value"]]]
            y = feat_df["Value"]
            model, cv_mae, cv_r2 = train_gbr_model(X, y)
            y_fit = model.predict(X)

            k1, k2, k3 = st.columns(3)
            k1.metric("In-sample MAE", amt_fmt(mean_absolute_error(y, y_fit)))
            k2.metric("In-sample R²", f"{r2_score(y, y_fit):.3f}")
            k3.metric("CV MAE (mean)", amt_fmt(cv_mae.mean()) if len(cv_mae)>0 else "—")

            plot_df = feat_df[["Year", "Value"]].copy()
            plot_df["Fitted"] = y_fit
            a_line = alt.Chart(plot_df).mark_line(color=ACCENT).encode(
                x="Year:T", y=alt.Y("Value:Q", title=f"{selected_indicator}")
            )
            f_line = alt.Chart(plot_df).mark_line(color=SECONDARY).encode(x="Year:T", y="Fitted:Q")
            st.altair_chart((a_line + f_line).properties(height=360), use_container_width=True)

            # Scenario controls
            st.markdown("Adjust recent dynamics to simulate policy effects (lower volatility, improved trajectory).")
            last_feat = feat_df.iloc[-1].copy()
            c1, c2 = st.columns(2)
            lag1 = c1.number_input("Last year value (lag1)", value=float(last_feat["Value_lag1"]), step=1000.0, format="%.0f")
            lag2 = c1.number_input("Value 2 years ago (lag2)", value=float(last_feat["Value_lag2"]), step=1000.0, format="%.0f")
            lag3 = c1.number_input("Value 3 years ago (lag3)", value=float(last_feat["Value_lag3"]), step=1000.0, format="%.0f")
            roll_mean_3 = c2.number_input("3-year rolling mean", value=float(last_feat["roll_mean_3"]), step=1000.0, format="%.0f")
            roll_std_3 = c2.number_input("3-year rolling std", value=float(last_feat["roll_std_3"]), step=1000.0, format="%.0f")
            yoy_pct = c2.number_input("YoY change (%)", value=float(last_feat["yoy_pct"]), step=1.0, format="%.2f")
            next_year = int(tidy["Year"].iloc[-1].year) + 1
            year_num = st.number_input("Forecast year", value=next_year, step=1)

            feature_cols = X.columns.tolist()
            scenario = pd.DataFrame([{
                "Value_lag1": lag1,
                "Value_lag2": lag2,
                "Value_lag3": lag3,
                "roll_mean_3": roll_mean_3,
                "roll_std_3": roll_std_3,
                "roll_mean_5": last_feat["roll_mean_5"],
                "roll_std_5": last_feat["roll_std_5"],
                "yoy_pct": yoy_pct,
                "year_num": year_num
            }])[feature_cols]

            pred_next = model.predict(scenario)[0]
            st.subheader("Scenario projection")
            st.metric("Projected next-year value", amt_fmt(pred_next))

            # Feature importance
            st.subheader("Feature importance")
            fi = pd.DataFrame({"Feature": feature_cols, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
            st.dataframe(fi.style.bar(subset=["Importance"], color=SECONDARY))
            bar = alt.Chart(fi).mark_bar(color=SECONDARY).encode(x=alt.X("Importance:Q"), y=alt.Y("Feature:N", sort="-x")).properties(height=300)
            st.altair_chart(bar, use_container_width=True)

# --------------------------------
# Compare tab (multi-country, normalized to index=100)
# --------------------------------
with tab_compare:
    if not compare_countries:
        st.info("Select one or more countries in the sidebar to compare.")
    else:
        comp = data[
            (data["Indicator"] == selected_indicator) &
            (data["Source"] == selected_source) &
            (data["Country"].isin(compare_countries))
        ].copy()
        if selected_unit and selected_unit != "(any)":
            comp = comp[comp["Unit"] == selected_unit]
        if selected_currency and selected_currency != "(any)":
            comp = comp[comp["Currency"] == selected_currency]
        if selected_freq and selected_freq != "(any)":
            comp = comp[comp["Frequency"] == selected_freq]

        # Aggregate per country/year
        comp = comp.groupby(["Country", "Year"])["Amount"].mean().reset_index()
        comp = comp.dropna(subset=["Year", "Amount"]).sort_values(["Country", "Year"])

        st.subheader(f"Comparison: {selected_indicator} by country (Index = 100 at base year)")
        # Normalize each country to index=100
        comp_list = []
        for c in compare_countries:
            d = comp[comp["Country"] == c].copy()
            d2, base_year = index_to_100(d.rename(columns={"Amount": "Value"}), "Value")
            if d2 is not None:
                d2["Country"] = c
                comp_list.append(d2[["Year", "Country", "Index"]])
        if not comp_list:
            st.info("Insufficient data to normalize series.")
        else:
            comp_idx = pd.concat(comp_list, axis=0)
            chart = alt.Chart(comp_idx).mark_line(point=True).encode(
                x="Year:T", y=alt.Y("Index:Q", title="Index (base=100)"), color="Country:N",
                tooltip=["Country", "Year", "Index"]
            ).properties(height=400)
            st.altair_chart(chart, use_container_width=True)
            st.caption("Each country normalized to the same base year for shape comparison; underlying values may differ by unit/currency.")

# --------------------------------
# Export tab
# --------------------------------
with tab_export:
    st.subheader("Export filtered series and forecasts")
    if tidy.empty:
        st.info("No filtered series to export.")
    else:
        csv_series = tidy.to_csv(index=False).encode("utf-8")
        st.download_button("Download filtered series (CSV)", csv_series, file_name=f"{selected_country}_{selected_indicator}_{selected_source}_series.csv", mime="text/csv")

        # Build forecast for export if available
        series = tidy.set_index("Year")["Value"]
        fc_frame, fc_err = arima_forecast(series, steps=5)
        if fc_err:
            st.info("Forecast unavailable for export: " + fc_err)
        else:
            last_year = series.index[-1].year
            future_years = pd.date_range(f"{last_year+1}-12-31", periods=5, freq="Y")
            fc_df = fc_frame.reset_index(drop=True)
            fc_df.insert(0, "Year", future_years)
            fc_df = fc_df.rename(columns={"mean": "Forecast", "mean_ci_lower": "CI Lower", "mean_ci_upper": "CI Upper"})
            csv_forecast = fc_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download forecast (CSV)", csv_forecast, file_name=f"{selected_country}_{selected_indicator}_{selected_source}_forecast.csv", mime="text/csv")

# --------------------------------
# Policy insights (SDG alignment)
# --------------------------------
st.markdown("---")
st.header("Policy insights (SDG alignment)")
st.markdown("""
- SDG 16 (Institutions): Use standardized, multi-source data pipelines like this to publish timely budget and macro execution dashboards; strengthen transparency and accountability.
- SDG 8 (Decent Work & Growth): Pair fiscal adjustments with growth-friendly measures (SME support, streamlined VAT refunds) to protect employment.
- SDG 9 (Industry, Infrastructure): Prioritize high-multiplier, shovel-ready projects and sequence investments to avoid cost bunching; monitor execution rates.
- SDG 10 (Reduced Inequality): If consolidation pressures household welfare, deploy targeted transfers and protect essential social spending.
- SDG 1 & 2 (No Poverty, Zero Hunger): For price-sensitive indicators, monitor food price dynamics and logistics constraints; consider targeted relief where needed.
- SDG 3 & SDG 4 (Health & Education): Track sectoral allocations and performance; aim for adequate per-capita support and efficiency via procurement transparency.
""")

st.caption("Built for multi-country, multi-indicator macrofiscal analysis with robust aggregation, forecasting, ML scenario simulation, normalization comparison, and exports.")
