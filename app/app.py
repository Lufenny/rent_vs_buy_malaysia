import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(page_title="Buy vs Rent — Full Workflow", layout="wide")

st.title("🏠 Buy vs Rent Malaysia — Full Workflow Dashboard")

# ------------------------------------------------------------
# 1. EXPECTED OUTCOMES
# ------------------------------------------------------------
st.header("🎯 Expected Outcomes")

st.markdown("""
- Clear buy vs rent insights for Malaysia  
- Multi-city comparison (KL, Penang, Johor)  
- Behavioral-adjusted investment modelling  
- Long-term Monte Carlo projections  
- Interactive Streamlit dashboard  
- Final cleaned dataset: **data.csv**
""")

st.markdown("---")

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
DATA_DIR = "../data"

files = {
    'Kuala Lumpur': f"{DATA_DIR}/kuala_lumpur_housing.csv",
    'Penang': f"{DATA_DIR}/penang_housing.csv",
    'Johor Bahru': f"{DATA_DIR}/johor_housing.csv",
}

@st.cache_data
def load_all():
    dfs = []
    for city, path in files.items():
        try:
            d = pd.read_csv(path, parse_dates=['date'])
            d['city'] = city
            dfs.append(d)
        except Exception as e:
            st.warning(f"Could not load {city}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

df_all = load_all()


# ------------------------------------------------------------
# SIDEBAR CONTROLS
# ------------------------------------------------------------
st.sidebar.header("Controls")
cities = st.sidebar.multiselect(
    "Select cities",
    options=df_all['city'].unique() if not df_all.empty else [],
    default=['Kuala Lumpur']
)

if not df_all.empty:
    date_range = st.sidebar.date_input(
        "Date range",
        [df_all['date'].min(), df_all['date'].max()]
    )

# ------------------------------------------------------------
# FILTER DATA
# ------------------------------------------------------------
df = df_all.copy()
if not df.empty and cities:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    df = df[(df['city'].isin(cities)) & (df['date'] >= start) & (df['date'] <= end)]
else:
    st.error("No data available.")


# ------------------------------------------------------------
# 2. EDA SECTION
# ------------------------------------------------------------
st.header("📊 1. Exploratory Data Analysis (EDA)")

# Time series
if 'property_index' in df.columns:
    st.subheader("Property Index Over Time")
    fig, ax = plt.subplots(figsize=(8,3))
    for city, g in df.groupby('city'):
        ax.plot(g['date'], g['property_index'], label=city)
    ax.legend()
    st.pyplot(fig)

# Distribution
st.subheader("Distribution of Monthly Returns")
if 'property_index' in df.columns:
    df['returns'] = df.groupby('city')['property_index'].pct_change()
    fig, ax = plt.subplots(figsize=(6,3))
    ax.hist(df['returns'].dropna(), bins=40)
    st.pyplot(fig)

# Outlier detection
st.subheader("Outlier Detection (Z-score)")
if 'property_index' in df.columns:
    z = np.abs(stats.zscore(df['property_index'].dropna()))
    outliers = df.iloc[np.where(z > 3)[0]]
    st.write(outliers if not outliers.empty else "No outliers detected.")

st.markdown("---")


# ------------------------------------------------------------
# 3. ANALYSIS
# ------------------------------------------------------------
st.header("📈 2. Analysis")

st.markdown("""
- Compare city growth  
- Evaluate EPF vs housing performance  
- Study OPR effects  
- Behavioral finance: savings & reinvestment discipline  
""")

if {'opr','property_index'}.issubset(df.columns):
    st.subheader("OPR vs Property Growth")
    df['year'] = df['date'].dt.year
    agg = df.groupby(['city','year']).agg({'property_index':'last','opr':'mean'})
    agg['growth'] = agg.groupby('city')['property_index'].pct_change()

    fig, ax = plt.subplots(figsize=(6,3))
    for city, g in agg.dropna().groupby('city'):
        ax.scatter(g['opr'], g['growth'], label=city)
    ax.legend()
    st.pyplot(fig)

st.markdown("---")


# ------------------------------------------------------------
# 4. DATA PROCESSING
# ------------------------------------------------------------
st.header("🧹 3. Data Processing")

st.markdown("""
- Merge multiple city datasets  
- Handle missing values  
- Convert dates  
- Engineering: monthly growth, real growth, reinvest probability  
""")

st.write(df.head())

st.markdown("---")


# ------------------------------------------------------------
# 5. MODELLING
# ------------------------------------------------------------
st.header("🧮 4. Modelling")

st.markdown("""
### Monte Carlo Model  
- Geometric Brownian motion  
- 10% volatility  
- Optional periodic crash  
- Reinvestment probability  
""")

st.sidebar.markdown("### Simulation Settings")
years = st.sidebar.slider("Years", 5, 40, 20)
sims = st.sidebar.slider("Simulations", 50, 1000, 200)
crash = st.sidebar.checkbox("Include crash every 5 years?", True)
reinvest_p = st.sidebar.slider("Reinvestment probability", 0.0, 1.0, 0.8)

# Simple Monte Carlo example
if not df.empty and 'property_index' in df.columns:
    last_price = df.sort_values('date')['property_index'].iloc[-1]
    returns = df['returns'].dropna()
    mu = returns.mean() * 12
    sigma = returns.std() * np.sqrt(12)

    T = years * 12
    dt = 1/12
    paths = np.zeros((sims, T))
    paths[:, 0] = last_price

    rng = np.random.default_rng(123)

    for t in range(1, T):
        z = rng.normal(size=sims)
        paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)

        if crash and t % (5*12) == 0:
            paths[:, t] *= 0.8

    median_path = np.median(paths, axis=0)

    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(median_path)
    st.pyplot(fig)

st.markdown("---")


# ------------------------------------------------------------
# 6. RESULTS INTERPRETATION
# ------------------------------------------------------------
st.header("📘 5. Results Interpretation")

st.markdown("""
- When buying is better  
- When renting + investing wins  
- Impact of OPR  
- City comparison  
- Behavioral influence  
""")

st.markdown("---")


# ------------------------------------------------------------
# 7. DEPLOYMENT
# ------------------------------------------------------------
st.header("🚀 6. Deployment")

st.markdown("""
- Streamlit interactive dashboard  
- CSV export  
- Scenario testing  
""")

st.markdown("---")


# ------------------------------------------------------------
# 8. FINAL DATASET EXPORT
# ------------------------------------------------------------
st.header("📁 7. Final Dataset (data.csv)")

if not df.empty:
    st.download_button(
        "Download data.csv",
        df.to_csv(index=False),
        file_name="data.csv",
        mime="text/csv"
    )
